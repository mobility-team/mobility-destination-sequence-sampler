//! Exact top-K validation oracle.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

use rayon::prelude::*;

use crate::errors::SamplerError;
use crate::input::{Context, Step};
use crate::model::{DestinationIndex, DestinationValue, OdGraph};
use crate::output::{OutputRow, OutputTable};
use crate::scoring::{
    adjusted_times, build_scoring_problem, fixed_destination_value, score_zones, Parameters,
    ScoringInputs, ScoringProblem, MIN_ACTIVITY_DURATION_HOURS,
};

struct OracleProblem<'a> {
    scoring: ScoringProblem,
    variable_by_layer: Vec<Option<usize>>,
    domains: Vec<&'a [usize]>,
    has_cross_home_anchor: bool,
}

/// Exhaustive exact-score distribution for a deliberately small context.
///
/// This is an analysis oracle, not a production sampler.  Callers must set an
/// explicit assignment cap before enumeration.
pub struct ExactDistribution {
    pub scores: Vec<f64>,
    pub assignment_lattice: u128,
}

#[allow(clippy::too_many_arguments)]
fn enumerate_distribution_assignments(
    variable: usize,
    assignments: &mut [usize],
    graph: &OdGraph,
    destinations: &DestinationIndex,
    context: &Context,
    problem: &OracleProblem<'_>,
    parameters: Parameters,
    scores: &mut Vec<f64>,
) {
    if variable == assignments.len() {
        let zones = context
            .steps
            .iter()
            .enumerate()
            .map(|(layer, step)| {
                step.fixed_destination
                    .map(|zone| graph.zone_index[&zone])
                    .unwrap_or_else(|| assignments[problem.variable_by_layer[layer].unwrap()])
            })
            .collect::<Vec<_>>();
        if let Some((score, _)) = score_zones(
            ScoringInputs {
                graph,
                destinations,
                context,
                problem: &problem.scoring,
                parameters,
            },
            &zones,
        ) {
            scores.push(score);
        }
        return;
    }
    for &zone in problem.domains[variable] {
        assignments[variable] = zone;
        enumerate_distribution_assignments(
            variable + 1,
            assignments,
            graph,
            destinations,
            context,
            problem,
            parameters,
            scores,
        );
    }
}

pub fn enumerate_reference_distribution(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    context: &Context,
    parameters: Parameters,
    max_assignments: usize,
) -> Result<ExactDistribution, SamplerError> {
    let (problem, _) = build_oracle_problem(graph, destinations, context)?;
    let assignment_lattice = problem.domains.iter().fold(1_u128, |count, domain| {
        count.saturating_mul(domain.len() as u128)
    });
    if assignment_lattice > max_assignments as u128 {
        return Err(SamplerError::InvalidInput(format!(
            "exact distribution needs {assignment_lattice} assignments in context {}; max_assignments={max_assignments}",
            context.context_id
        )));
    }
    let mut scores = Vec::with_capacity(assignment_lattice as usize);
    let mut assignments = vec![0; problem.domains.len()];
    enumerate_distribution_assignments(
        0,
        &mut assignments,
        graph,
        destinations,
        context,
        &problem,
        parameters,
        &mut scores,
    );
    if scores.is_empty() {
        return Err(SamplerError::NoFeasibleSequence {
            context_id: context.context_id,
            origin: context.initial_zone,
        });
    }
    scores.sort_unstable_by(|left, right| right.total_cmp(left));
    Ok(ExactDistribution {
        scores,
        assignment_lattice,
    })
}

fn split_ranges_at_home(graph: &OdGraph, context: &Context) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut start = 0;
    for (layer, step) in context.steps.iter().enumerate() {
        if step.fixed_destination == Some(context.initial_zone) {
            ranges.push((start, layer + 1));
            start = layer + 1;
        }
    }
    if start < context.steps.len() {
        ranges.push((start, context.steps.len()));
    }
    if ranges.is_empty() {
        ranges.push((0, context.steps.len()));
    }
    debug_assert!(graph.zone_index.contains_key(&context.initial_zone));
    ranges
}

fn build_oracle_problem<'a>(
    graph: &OdGraph,
    destinations: &'a DestinationIndex,
    context: &Context,
) -> Result<(OracleProblem<'a>, Vec<(usize, usize)>), SamplerError> {
    let scoring = build_scoring_problem(context)?;
    let ranges = split_ranges_at_home(graph, context);
    let mut segment_by_layer = vec![0usize; context.steps.len()];
    for (segment, &(start, end)) in ranges.iter().enumerate() {
        segment_by_layer[start..end].fill(segment);
    }

    let mut anchor_variables = BTreeMap::new();
    let mut anchor_segment_by_id = BTreeMap::new();
    let mut has_cross_home_anchor = false;
    let mut variable_activities = Vec::new();
    let mut variable_by_layer = Vec::with_capacity(context.steps.len());
    for (layer, step) in context.steps.iter().enumerate() {
        if step.fixed_destination.is_some() {
            variable_by_layer.push(None);
            continue;
        }
        let variable = if let Some(anchor_id) = step.anchor_id {
            if let Some(&variable) = anchor_variables.get(&anchor_id) {
                if anchor_segment_by_id[&anchor_id] != segment_by_layer[layer] {
                    has_cross_home_anchor = true;
                }
                variable
            } else {
                let variable = variable_activities.len();
                variable_activities.push(step.activity_id);
                anchor_variables.insert(anchor_id, variable);
                anchor_segment_by_id.insert(anchor_id, segment_by_layer[layer]);
                variable
            }
        } else {
            let variable = variable_activities.len();
            variable_activities.push(step.activity_id);
            variable
        };
        variable_by_layer.push(Some(variable));
    }
    let domains = variable_activities
        .iter()
        .map(|&activity_id| {
            destinations
                .domain(activity_id)
                .filter(|domain| !domain.is_empty())
                .ok_or(SamplerError::NoFeasibleSequence {
                    context_id: context.context_id,
                    origin: context.initial_zone,
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((
        OracleProblem {
            scoring,
            variable_by_layer,
            domains,
            has_cross_home_anchor,
        },
        ranges,
    ))
}
#[derive(Clone)]
struct HeapState {
    upper_bound: f64,
    score: f64,
    layer: usize,
    current_zone: usize,
    adjusted_arrival: Option<f64>,
    path_index: Option<usize>,
    anchor_destinations: BTreeMap<u32, usize>,
    insertion_order: u64,
}

struct PathNode {
    parent: Option<usize>,
    zone: usize,
}

#[derive(Clone)]
struct PendingChild {
    upper_bound: f64,
    score: f64,
    destination: usize,
    adjusted_arrival: f64,
}

struct SiblingState {
    parent: HeapState,
    children: Vec<PendingChild>,
}

enum SearchEntry {
    State(HeapState),
    Siblings {
        upper_bound: f64,
        insertion_order: u64,
        siblings: SiblingState,
    },
}

impl SearchEntry {
    fn upper_bound(&self) -> f64 {
        match self {
            Self::State(state) => state.upper_bound,
            Self::Siblings { upper_bound, .. } => *upper_bound,
        }
    }

    fn insertion_order(&self) -> u64 {
        match self {
            Self::State(state) => state.insertion_order,
            Self::Siblings {
                insertion_order, ..
            } => *insertion_order,
        }
    }
}

impl PartialEq for SearchEntry {
    fn eq(&self, other: &Self) -> bool {
        self.upper_bound() == other.upper_bound()
            && self.insertion_order() == other.insertion_order()
    }
}

impl Eq for SearchEntry {}

impl PartialOrd for SearchEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.upper_bound()
            .total_cmp(&other.upper_bound())
            .then_with(|| other.insertion_order().cmp(&self.insertion_order()))
    }
}

#[derive(Debug, Default)]
pub struct HeapSearchReport {
    pub contexts: u64,
    pub split_contexts: u64,
    pub conditioned_anchor_contexts: u64,
    pub anchor_conditions_considered: u64,
    pub anchor_conditions_pruned: u64,
    pub incumbent_contexts: u64,
    pub incumbent_children_considered: u64,
    pub children_pruned_by_incumbent: u64,
    pub queue_entries_popped: u64,
    pub sibling_entries_popped: u64,
    pub states_popped: u64,
    pub states_pushed: u64,
    pub children_considered: u64,
    pub complete_plans: u64,
    pub maximum_heap_size: u64,
    pub assignment_lattice: u128,
}

impl HeapSearchReport {
    fn add_subsearch(&mut self, other: &Self) {
        self.split_contexts += other.split_contexts;
        self.conditioned_anchor_contexts += other.conditioned_anchor_contexts;
        self.anchor_conditions_considered += other.anchor_conditions_considered;
        self.anchor_conditions_pruned += other.anchor_conditions_pruned;
        self.incumbent_contexts += other.incumbent_contexts;
        self.incumbent_children_considered += other.incumbent_children_considered;
        self.children_pruned_by_incumbent += other.children_pruned_by_incumbent;
        self.queue_entries_popped += other.queue_entries_popped;
        self.sibling_entries_popped += other.sibling_entries_popped;
        self.states_popped += other.states_popped;
        self.states_pushed += other.states_pushed;
        self.children_considered += other.children_considered;
        self.complete_plans += other.complete_plans;
        self.maximum_heap_size = self.maximum_heap_size.max(other.maximum_heap_size);
    }

    fn add_context_result(&mut self, other: &Self) {
        self.contexts += other.contexts;
        self.add_subsearch(other);
        self.assignment_lattice = self
            .assignment_lattice
            .saturating_add(other.assignment_lattice);
    }
}

#[derive(Clone)]
struct RankedPlan {
    zones: Vec<usize>,
    score: f64,
}

fn independent_home_ranges(
    context: &Context,
    ranges: &[(usize, usize)],
) -> Option<Vec<(usize, usize)>> {
    if ranges.len() <= 1 {
        return None;
    }
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for &(start, end) in ranges {
        let only_fixed = context.steps[start..end]
            .iter()
            .all(|step| step.fixed_destination.is_some());
        if only_fixed && !merged.is_empty() {
            merged.last_mut().unwrap().1 = end;
        } else {
            merged.push((start, end));
        }
    }
    if merged.len() <= 1
        || merged[..merged.len() - 1].iter().any(|&(_, end)| {
            context.steps[end].fixed_destination.is_none()
                && context.steps[end]
                    .arrival_time_rigidity
                    .is_none_or(|rigidity| rigidity != 0.0)
        })
    {
        None
    } else {
        Some(merged)
    }
}

fn cross_home_anchor_ids(context: &Context, ranges: &[(usize, usize)]) -> Vec<u32> {
    let mut segment_by_layer = vec![0usize; context.steps.len()];
    for (segment, &(start, end)) in ranges.iter().enumerate() {
        segment_by_layer[start..end].fill(segment);
    }
    let mut first_segment_by_anchor = BTreeMap::new();
    let mut cross_home_anchors = BTreeSet::new();
    for (layer, step) in context.steps.iter().enumerate() {
        let Some(anchor_id) = step.anchor_id else {
            continue;
        };
        match first_segment_by_anchor.entry(anchor_id) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(segment_by_layer[layer]);
            }
            std::collections::btree_map::Entry::Occupied(entry)
                if *entry.get() != segment_by_layer[layer] =>
            {
                cross_home_anchors.insert(anchor_id);
            }
            _ => {}
        }
    }
    cross_home_anchors.into_iter().collect()
}

fn ranked_plans_from_output(
    graph: &OdGraph,
    table: &OutputTable,
    n_steps: usize,
) -> Vec<RankedPlan> {
    (0..table.destination.len())
        .step_by(n_steps)
        .map(|start| RankedPlan {
            zones: table.destination[start..start + n_steps]
                .iter()
                .map(|zone_id| graph.zone_index[zone_id])
                .collect(),
            score: table.total_log_weight[start],
        })
        .collect()
}

fn merge_ranked_segments(segments: &[Vec<RankedPlan>], k: usize) -> Vec<Vec<usize>> {
    let mut combinations = vec![RankedPlan {
        zones: Vec::new(),
        score: 0.0,
    }];
    for segment in segments {
        let mut next = Vec::with_capacity(k.saturating_mul(k));
        for prefix in &combinations {
            for plan in segment {
                let mut zones = Vec::with_capacity(prefix.zones.len() + plan.zones.len());
                zones.extend_from_slice(&prefix.zones);
                zones.extend_from_slice(&plan.zones);
                next.push(RankedPlan {
                    zones,
                    score: prefix.score + plan.score,
                });
            }
        }
        next.sort_unstable_by(|left, right| right.score.total_cmp(&left.score));
        next.truncate(k);
        combinations = next;
    }
    combinations.into_iter().map(|plan| plan.zones).collect()
}

fn append_ranked_plans(
    output: &mut OutputTable,
    graph: &OdGraph,
    destinations: &DestinationIndex,
    context: &Context,
    problem: &OracleProblem<'_>,
    plans: Vec<Vec<usize>>,
    parameters: Parameters,
) -> Result<(), SamplerError> {
    let mut scored = Vec::with_capacity(plans.len());
    for zones in plans {
        let Some((score, local_weights)) = score_zones(
            ScoringInputs {
                graph,
                destinations,
                context,
                problem: &problem.scoring,
                parameters,
            },
            &zones,
        ) else {
            return Err(SamplerError::InvalidInput(format!(
                "heap reference search returned an infeasible complete plan in context {}",
                context.context_id
            )));
        };
        scored.push((zones, score, local_weights));
    }
    scored.sort_unstable_by(
        |(left_zones, left_score, _), (right_zones, right_score, _)| {
            right_score
                .total_cmp(left_score)
                .then_with(|| left_zones.cmp(right_zones))
        },
    );
    for (rank, (zones, _, local_weights)) in scored.into_iter().enumerate() {
        let mut suffix = 0.0;
        let mut suffix_values = vec![0.0; local_weights.len()];
        for layer in (0..local_weights.len()).rev() {
            suffix += local_weights[layer];
            suffix_values[layer] = suffix;
        }
        let mut origin = graph.zone_index[&context.initial_zone];
        for layer in 0..context.steps.len() {
            let destination = zones[layer];
            output.push(OutputRow {
                context_id: context.context_id,
                draw_id: rank as u32 + 1,
                layer: layer as u32,
                origin: graph.zone_ids[origin],
                destination: graph.zone_ids[destination],
                local_log_weight: local_weights[layer],
                total_log_weight: suffix_values[layer],
            });
            origin = destination;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn append_two_step_plans(
    output: &mut OutputTable,
    report: &mut HeapSearchReport,
    graph: &OdGraph,
    destinations: &DestinationIndex,
    context: &Context,
    problem: &OracleProblem<'_>,
    parameters: Parameters,
    k: usize,
) -> Result<bool, SamplerError> {
    let terminal_zone = context.steps[1].fixed_destination.ok_or_else(|| {
        SamplerError::InvalidInput(format!(
            "context {} needs a fixed terminal destination for exact top-K search",
            context.context_id
        ))
    })?;
    let terminal = *graph.zone_index.get(&terminal_zone).ok_or_else(|| {
        SamplerError::InvalidInput(format!(
            "context {} terminal destination {} is absent from the OD graph",
            context.context_id, terminal_zone
        ))
    })?;
    let first = context.steps[0];
    let candidates = if let Some(fixed) = first.fixed_destination {
        vec![*graph.zone_index.get(&fixed).ok_or_else(|| {
            SamplerError::InvalidInput(format!(
                "context {} fixed destination {} is absent from the OD graph",
                context.context_id, fixed
            ))
        })?]
    } else {
        destinations
            .domain(first.activity_id)
            .ok_or(SamplerError::NoFeasibleSequence {
                context_id: context.context_id,
                origin: context.initial_zone,
            })?
            .to_vec()
    };
    let mut plans = Vec::with_capacity(candidates.len());
    for destination in candidates {
        report.children_considered += 1;
        let zones = vec![destination, terminal];
        if let Some((score, _)) = score_zones(
            ScoringInputs {
                graph,
                destinations,
                context,
                problem: &problem.scoring,
                parameters,
            },
            &zones,
        ) {
            plans.push((score, zones));
        }
    }
    if plans.is_empty() {
        return Ok(false);
    }
    report.complete_plans += plans.len() as u64;
    plans.sort_unstable_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    append_ranked_plans(
        output,
        graph,
        destinations,
        context,
        problem,
        plans.into_iter().take(k).map(|(_, zones)| zones).collect(),
        parameters,
    )?;
    Ok(true)
}

fn materialize_child(
    context: &Context,
    parent: &HeapState,
    pending: PendingChild,
    path: &mut Vec<PathNode>,
    insertion_order: u64,
) -> HeapState {
    let mut child = parent.clone();
    child.upper_bound = pending.upper_bound;
    child.score = pending.score;
    child.layer += 1;
    child.current_zone = pending.destination;
    child.adjusted_arrival = Some(pending.adjusted_arrival);
    child.insertion_order = insertion_order;
    path.push(PathNode {
        parent: parent.path_index,
        zone: pending.destination,
    });
    child.path_index = Some(path.len() - 1);
    if let Some(anchor_id) = context.steps[parent.layer].anchor_id {
        child
            .anchor_destinations
            .entry(anchor_id)
            .or_insert(pending.destination);
    }
    child
}

fn duration_upper_bound(context: &Context, layer: usize, parameters: Parameters) -> f64 {
    let step = context.steps[layer];
    if !parameters.update_plan_timings {
        return step.duration_per_person.max(MIN_ACTIVITY_DURATION_HOURS);
    }
    let arrival = step
        .arrival_time
        .expect("reference contexts have arrival times");
    let rigidity = step
        .arrival_time_rigidity
        .expect("reference contexts have rigidity");
    let reference_time = (arrival - step.departure_time).clamp(0.0, 24.0);
    let earliest_arrival = arrival - (1.0 - rigidity) * reference_time;
    let latest_next_departure = if let Some(next_step) = context.steps.get(layer + 1) {
        let next_arrival = next_step
            .arrival_time
            .expect("reference contexts have arrival times");
        let next_rigidity = next_step
            .arrival_time_rigidity
            .expect("reference contexts have rigidity");
        let next_reference_time = (next_arrival - next_step.departure_time).clamp(0.0, 24.0);
        next_step.departure_time + next_rigidity * next_reference_time
    } else {
        step.next_departure_time
    };
    (latest_next_departure - earliest_arrival).max(MIN_ACTIVITY_DURATION_HOURS)
}

fn activity_coefficient(destination: DestinationValue, step: Step, parameters: Parameters) -> f64 {
    if parameters.use_shadow_prices {
        destination.country_value_coefficient * step.value_of_time + destination.shadow_price
    } else {
        destination.country_value_coefficient * destination.saturation_utility * step.value_of_time
    }
}

#[derive(Clone, Copy)]
struct LayerUpperBound {
    maximum_activity_coefficient: f64,
    attraction: f64,
    duration: f64,
}

fn build_layer_upper_bounds(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    context: &Context,
    problem: &OracleProblem<'_>,
    parameters: Parameters,
) -> Vec<LayerUpperBound> {
    context
        .steps
        .iter()
        .enumerate()
        .map(|(layer, &step)| {
            let maximum_activity_coefficient = if let Some(zone_id) = step.fixed_destination {
                activity_coefficient(
                    fixed_destination_value(
                        destinations.activity(step.activity_id),
                        graph.zone_index[&zone_id],
                    ),
                    step,
                    parameters,
                )
            } else {
                problem.domains[problem.variable_by_layer[layer].unwrap()]
                    .iter()
                    .filter_map(|&zone| destinations.activity(step.activity_id).unwrap()[zone])
                    .map(|value| activity_coefficient(value, step, parameters))
                    .fold(f64::NEG_INFINITY, f64::max)
            };
            LayerUpperBound {
                maximum_activity_coefficient,
                attraction: attraction_upper_bound(destinations, context, problem, layer),
                duration: duration_upper_bound(context, layer, parameters),
            }
        })
        .collect()
}

fn activity_utility_upper_bound(
    destinations: &DestinationIndex,
    context: &Context,
    state: &HeapState,
    layer: usize,
    duration: f64,
    parameters: Parameters,
    layer_bound: LayerUpperBound,
) -> f64 {
    let step = context.steps[layer];
    let coefficient = if layer + 1 == state.layer {
        activity_coefficient(
            fixed_destination_value(destinations.activity(step.activity_id), state.current_zone),
            step,
            parameters,
        )
    } else if let Some(anchor_id) = step.anchor_id {
        state.anchor_destinations.get(&anchor_id).map_or(
            layer_bound.maximum_activity_coefficient,
            |&zone| {
                activity_coefficient(
                    destinations.activity(step.activity_id).unwrap()[zone].unwrap(),
                    step,
                    parameters,
                )
            },
        )
    } else {
        layer_bound.maximum_activity_coefficient
    };
    let duration_factor = (duration / step.min_activity_time).ln().max(0.0);
    (coefficient * step.mean_duration_per_person * duration_factor).max(0.0)
}

fn relaxed_candidates(
    graph: &OdGraph,
    context: &Context,
    problem: &OracleProblem<'_>,
    layer: usize,
) -> Vec<usize> {
    context.steps[layer].fixed_destination.map_or_else(
        || problem.domains[problem.variable_by_layer[layer].unwrap()].to_vec(),
        |zone_id| vec![graph.zone_index[&zone_id]],
    )
}

fn build_relaxed_suffix_bounds(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    context: &Context,
    problem: &OracleProblem<'_>,
    parameters: Parameters,
    layer_bounds: &[LayerUpperBound],
) -> Vec<Vec<f64>> {
    let candidates: Vec<Vec<usize>> = (0..context.steps.len())
        .map(|layer| relaxed_candidates(graph, context, problem, layer))
        .collect();
    let mut suffix = vec![vec![f64::NEG_INFINITY; graph.zone_ids.len()]; context.steps.len()];

    for layer in (0..context.steps.len()).rev() {
        let origins = if layer == 0 {
            std::slice::from_ref(&graph.zone_index[&context.initial_zone])
        } else {
            candidates[layer - 1].as_slice()
        };
        let step = context.steps[layer];
        let duration_factor = (layer_bounds[layer].duration / step.min_activity_time)
            .ln()
            .max(0.0);
        for &origin in origins {
            let mut maximum = f64::NEG_INFINITY;
            for &destination in &candidates[layer] {
                let Some(edge) = graph.edge_to(origin, destination) else {
                    continue;
                };
                let value =
                    fixed_destination_value(destinations.activity(step.activity_id), destination);
                let attraction = if step.fixed_destination.is_some()
                    || !problem.scoring.is_first_choice(layer)
                {
                    0.0
                } else {
                    value.log_opportunity_capacity
                };
                let activity_utility = (activity_coefficient(value, step, parameters)
                    * step.mean_duration_per_person
                    * duration_factor)
                    .max(0.0);
                let continuation = if layer + 1 == context.steps.len() {
                    0.0
                } else {
                    suffix[layer + 1][destination]
                };
                maximum = maximum.max(
                    attraction - parameters.logit_scale * edge.cost
                        + parameters.logit_scale * activity_utility
                        + continuation,
                );
            }
            suffix[layer][origin] = maximum;
        }
    }
    suffix
}

fn attraction_upper_bound(
    destinations: &DestinationIndex,
    context: &Context,
    problem: &OracleProblem<'_>,
    layer: usize,
) -> f64 {
    let step = context.steps[layer];
    if step.fixed_destination.is_some() || !problem.scoring.is_first_choice(layer) {
        return 0.0;
    }
    problem.domains[problem.variable_by_layer[layer].unwrap()]
        .iter()
        .filter_map(|&zone| destinations.activity(step.activity_id).unwrap()[zone])
        .map(|value| value.log_opportunity_capacity)
        .fold(f64::NEG_INFINITY, f64::max)
}

#[allow(clippy::too_many_arguments)]
fn anchor_condition_upper_bound(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    context: &Context,
    problem: &OracleProblem<'_>,
    anchor_id: u32,
    anchor_zone: usize,
    parameters: Parameters,
    minimum_edge_cost: f64,
    layer_bounds: &[LayerUpperBound],
) -> f64 {
    let mut bound = 0.0;
    for (layer, &step) in context.steps.iter().enumerate() {
        let is_conditioned_anchor = step.anchor_id == Some(anchor_id);
        let destination_value = is_conditioned_anchor
            .then(|| destinations.activity(step.activity_id).unwrap()[anchor_zone].unwrap());
        let coefficient = destination_value
            .map_or(layer_bounds[layer].maximum_activity_coefficient, |value| {
                activity_coefficient(value, step, parameters)
            });
        let duration_factor = (layer_bounds[layer].duration / step.min_activity_time)
            .ln()
            .max(0.0);
        bound += parameters.logit_scale
            * (coefficient * step.mean_duration_per_person * duration_factor).max(0.0);
        bound += if is_conditioned_anchor && problem.scoring.is_first_choice(layer) {
            destination_value.unwrap().log_opportunity_capacity
        } else if is_conditioned_anchor {
            0.0
        } else {
            layer_bounds[layer].attraction
        };

        let previous_step = layer.checked_sub(1).map(|index| context.steps[index]);
        let fixed_origin = previous_step.map_or_else(
            || Some(graph.zone_index[&context.initial_zone]),
            |previous| {
                if previous.anchor_id == Some(anchor_id) {
                    Some(anchor_zone)
                } else {
                    previous
                        .fixed_destination
                        .map(|zone_id| graph.zone_index[&zone_id])
                }
            },
        );
        let fixed_destination = if is_conditioned_anchor {
            Some(anchor_zone)
        } else {
            step.fixed_destination
                .map(|zone_id| graph.zone_index[&zone_id])
        };
        let leg_minimum = match (fixed_origin, fixed_destination) {
            (Some(origin), Some(destination)) => graph
                .edge_to(origin, destination)
                .map_or(f64::INFINITY, |edge| edge.cost),
            (Some(origin), None) => {
                let domain = problem.domains[problem.variable_by_layer[layer].unwrap()];
                domain
                    .iter()
                    .filter_map(|&destination| graph.edge_to(origin, destination))
                    .map(|edge| edge.cost)
                    .fold(f64::INFINITY, f64::min)
            }
            (None, Some(destination)) => {
                let previous_layer = layer - 1;
                let domain = problem.domains[problem.variable_by_layer[previous_layer].unwrap()];
                domain
                    .iter()
                    .filter_map(|&origin| graph.edge_to(origin, destination))
                    .map(|edge| edge.cost)
                    .fold(f64::INFINITY, f64::min)
            }
            (None, None) => minimum_edge_cost,
        };
        bound -= parameters.logit_scale * leg_minimum;
    }
    bound
}

fn state_upper_bound(
    destinations: &DestinationIndex,
    context: &Context,
    state: &HeapState,
    parameters: Parameters,
    minimum_edge_cost: f64,
    layer_bounds: &[LayerUpperBound],
    relaxed_suffix: Option<&[Vec<f64>]>,
) -> f64 {
    let chosen = state.layer;
    if let Some(suffix) = relaxed_suffix {
        if chosen == 0 {
            return state.score + suffix[0][state.current_zone];
        }
        let layer = chosen - 1;
        let duration = if parameters.update_plan_timings {
            let latest_next_departure = if let Some(next_step) = context.steps.get(chosen) {
                let next_arrival = next_step.arrival_time.unwrap();
                let next_rigidity = next_step.arrival_time_rigidity.unwrap();
                let next_reference = (next_arrival - next_step.departure_time).clamp(0.0, 24.0);
                next_step.departure_time + next_rigidity * next_reference
            } else {
                context.steps[layer].next_departure_time
            };
            (latest_next_departure - state.adjusted_arrival.unwrap())
                .max(MIN_ACTIVITY_DURATION_HOURS)
        } else {
            context.steps[layer].duration_per_person
        };
        return state.score
            + parameters.logit_scale
                * activity_utility_upper_bound(
                    destinations,
                    context,
                    state,
                    layer,
                    duration,
                    parameters,
                    layer_bounds[layer],
                )
            + suffix[chosen][state.current_zone];
    }
    let mut bound = state.score;

    if chosen > 0 {
        let layer = chosen - 1;
        let duration = if parameters.update_plan_timings {
            let latest_next_departure = if let Some(next_step) = context.steps.get(chosen) {
                let next_arrival = next_step.arrival_time.unwrap();
                let next_rigidity = next_step.arrival_time_rigidity.unwrap();
                let next_reference = (next_arrival - next_step.departure_time).clamp(0.0, 24.0);
                next_step.departure_time + next_rigidity * next_reference
            } else {
                context.steps[layer].next_departure_time
            };
            (latest_next_departure - state.adjusted_arrival.unwrap())
                .max(MIN_ACTIVITY_DURATION_HOURS)
        } else {
            context.steps[layer].duration_per_person
        };
        bound += parameters.logit_scale
            * activity_utility_upper_bound(
                destinations,
                context,
                state,
                layer,
                duration,
                parameters,
                layer_bounds[layer],
            );
    }

    for (layer, &layer_bound) in layer_bounds.iter().enumerate().skip(chosen) {
        bound += layer_bound.attraction;
        bound -= parameters.logit_scale * minimum_edge_cost;
        bound += parameters.logit_scale
            * activity_utility_upper_bound(
                destinations,
                context,
                state,
                layer,
                layer_bound.duration,
                parameters,
                layer_bound,
            );
    }
    bound
}

fn exact_activity_utility(
    destinations: &DestinationIndex,
    context: &Context,
    layer: usize,
    destination: usize,
    duration: f64,
    parameters: Parameters,
) -> f64 {
    let step = context.steps[layer];
    let value = fixed_destination_value(destinations.activity(step.activity_id), destination);
    let duration_factor = (duration / step.min_activity_time).ln().max(0.0);
    activity_coefficient(value, step, parameters) * step.mean_duration_per_person * duration_factor
}

fn heap_candidates<'a>(
    graph: &OdGraph,
    context: &Context,
    problem: &'a OracleProblem<'a>,
    state: &'a HeapState,
    layer: usize,
) -> Result<&'a [usize], usize> {
    let step = context.steps[layer];
    if let Some(zone_id) = step.fixed_destination {
        return Err(graph.zone_index[&zone_id]);
    }
    if let Some(anchor_id) = step.anchor_id {
        if let Some(&zone) = state.anchor_destinations.get(&anchor_id) {
            return Err(zone);
        }
    }
    Ok(problem.domains[problem.variable_by_layer[layer].unwrap()])
}

#[allow(clippy::too_many_arguments)]
fn pending_children_for_state(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    context: &Context,
    problem: &OracleProblem<'_>,
    state: &HeapState,
    parameters: Parameters,
    minimum_edge_cost: f64,
    layer_bounds: &[LayerUpperBound],
    relaxed_suffix: Option<&[Vec<f64>]>,
) -> (Vec<PendingChild>, u64) {
    let layer = state.layer;
    let candidate_result = heap_candidates(graph, context, problem, state, layer);
    let singleton;
    let candidates = match candidate_result {
        Ok(candidates) => candidates,
        Err(destination) => {
            singleton = [destination];
            &singleton
        }
    };
    let origin = state.current_zone;
    let mut pending_children = Vec::with_capacity(candidates.len());

    for &destination in candidates {
        let Some(edge) = graph.edge_to(origin, destination) else {
            continue;
        };
        let step = context.steps[layer];
        let destination_value = if step.fixed_destination.is_some() {
            fixed_destination_value(destinations.activity(step.activity_id), destination)
        } else {
            let Some(value) = destinations.activity(step.activity_id).unwrap()[destination] else {
                continue;
            };
            value
        };
        let Some((adjusted_departure, adjusted_arrival)) = adjusted_times(step, edge) else {
            continue;
        };

        let mut child = state.clone();
        child.layer += 1;
        child.current_zone = destination;
        child.adjusted_arrival = Some(adjusted_arrival);
        if let Some(anchor_id) = step.anchor_id {
            child
                .anchor_destinations
                .entry(anchor_id)
                .or_insert(destination);
        }
        if layer > 0 {
            let previous_duration = if parameters.update_plan_timings {
                adjusted_departure - state.adjusted_arrival.unwrap()
            } else {
                context.steps[layer - 1].duration_per_person
            };
            if parameters.update_plan_timings && previous_duration <= 0.0 {
                continue;
            }
            child.score += parameters.logit_scale
                * exact_activity_utility(
                    destinations,
                    context,
                    layer - 1,
                    state.current_zone,
                    previous_duration,
                    parameters,
                );
        }
        let attraction =
            if step.fixed_destination.is_some() || !problem.scoring.is_first_choice(layer) {
                0.0
            } else {
                destination_value.log_opportunity_capacity
            };
        child.score += attraction - parameters.logit_scale * edge.cost;

        if layer + 1 == context.steps.len() {
            let duration = if step.fixed_destination.is_some() {
                MIN_ACTIVITY_DURATION_HOURS
            } else if parameters.update_plan_timings {
                continue;
            } else {
                step.duration_per_person
            };
            child.score += parameters.logit_scale
                * exact_activity_utility(
                    destinations,
                    context,
                    layer,
                    destination,
                    duration,
                    parameters,
                );
            child.upper_bound = child.score;
        } else {
            child.upper_bound = state_upper_bound(
                destinations,
                context,
                &child,
                parameters,
                minimum_edge_cost,
                layer_bounds,
                relaxed_suffix,
            );
        }
        pending_children.push(PendingChild {
            upper_bound: child.upper_bound,
            score: child.score,
            destination,
            adjusted_arrival,
        });
    }
    (pending_children, candidates.len() as u64)
}

#[allow(clippy::too_many_arguments)]
fn greedy_incumbent(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    context: &Context,
    problem: &OracleProblem<'_>,
    initial: &HeapState,
    parameters: Parameters,
    minimum_edge_cost: f64,
    layer_bounds: &[LayerUpperBound],
    relaxed_suffix: Option<&[Vec<f64>]>,
) -> (Option<RankedPlan>, u64) {
    let mut state = initial.clone();
    let mut zones = Vec::with_capacity(context.steps.len());
    let mut children_considered = 0;

    while state.layer < context.steps.len() {
        let (children, considered) = pending_children_for_state(
            graph,
            destinations,
            context,
            problem,
            &state,
            parameters,
            minimum_edge_cost,
            layer_bounds,
            relaxed_suffix,
        );
        children_considered += considered;
        let Some(best) = children.into_iter().max_by(|left, right| {
            left.upper_bound
                .total_cmp(&right.upper_bound)
                .then_with(|| left.destination.cmp(&right.destination))
        }) else {
            return (None, children_considered);
        };
        let step = context.steps[state.layer];
        state.upper_bound = best.upper_bound;
        state.score = best.score;
        state.layer += 1;
        state.current_zone = best.destination;
        state.adjusted_arrival = Some(best.adjusted_arrival);
        if let Some(anchor_id) = step.anchor_id {
            state
                .anchor_destinations
                .entry(anchor_id)
                .or_insert(best.destination);
        }
        zones.push(best.destination);
    }
    (
        Some(RankedPlan {
            zones,
            score: state.score,
        }),
        children_considered,
    )
}

fn search_reference_top_k_sequential(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    contexts: &[Context],
    parameters: Parameters,
    k: usize,
    max_states: usize,
) -> Result<(OutputTable, HeapSearchReport), SamplerError> {
    let minimum_edge_cost = graph
        .edges
        .iter()
        .map(|edge| edge.cost)
        .fold(f64::INFINITY, f64::min);
    let mut output = OutputTable::default();
    let mut report = HeapSearchReport::default();

    for context in contexts {
        report.contexts += 1;
        let (problem, ranges) = build_oracle_problem(graph, destinations, context)?;
        let lattice = problem.domains.iter().fold(1u128, |count, domain| {
            count.saturating_mul(domain.len() as u128)
        });
        report.assignment_lattice = report.assignment_lattice.saturating_add(lattice);

        if context.steps.len() == 2 {
            if append_two_step_plans(
                &mut output,
                &mut report,
                graph,
                destinations,
                context,
                &problem,
                parameters,
                k,
            )? {
                continue;
            }
            if parameters.skip_infeasible {
                continue;
            }
            return Err(SamplerError::NoFeasibleSequence {
                context_id: context.context_id,
                origin: context.initial_zone,
            });
        }

        if !problem.has_cross_home_anchor {
            if let Some(independent_ranges) = independent_home_ranges(context, &ranges) {
                let mut segment_plans = Vec::with_capacity(independent_ranges.len());
                for &(start, end) in &independent_ranges {
                    let mut steps = context.steps[start..end].to_vec();
                    for (layer, step) in steps.iter_mut().enumerate() {
                        step.layer = layer as u32;
                    }
                    let segment_context = Context {
                        context_id: context.context_id,
                        initial_zone: context.initial_zone,
                        steps,
                    };
                    let (segment_problem, _) =
                        build_oracle_problem(graph, destinations, &segment_context)?;
                    let segment_k = segment_problem
                        .domains
                        .iter()
                        .try_fold(1usize, |count, domain| count.checked_mul(domain.len()))
                        .unwrap_or(usize::MAX)
                        .min(k);
                    let (table, segment_report) = search_reference_top_k_sequential(
                        graph,
                        destinations,
                        std::slice::from_ref(&segment_context),
                        parameters,
                        segment_k,
                        max_states,
                    )?;
                    segment_plans.push(ranked_plans_from_output(
                        graph,
                        &table,
                        segment_context.steps.len(),
                    ));
                    report.add_subsearch(&segment_report);
                }
                report.split_contexts += 1;
                append_ranked_plans(
                    &mut output,
                    graph,
                    destinations,
                    context,
                    &problem,
                    merge_ranked_segments(&segment_plans, k),
                    parameters,
                )?;
                continue;
            }
        }

        let layer_bounds =
            build_layer_upper_bounds(graph, destinations, context, &problem, parameters);
        let relaxed_suffix = (problem.domains.len() >= 2).then(|| {
            build_relaxed_suffix_bounds(
                graph,
                destinations,
                context,
                &problem,
                parameters,
                &layer_bounds,
            )
        });

        let cross_home_anchors = cross_home_anchor_ids(context, &ranges);
        if k == 1 && cross_home_anchors.len() == 1 {
            let anchor_id = cross_home_anchors[0];
            let anchor_step = context
                .steps
                .iter()
                .find(|step| step.anchor_id == Some(anchor_id))
                .unwrap();
            let anchor_domain = destinations.domain(anchor_step.activity_id).unwrap();
            let mut initial = HeapState {
                upper_bound: f64::INFINITY,
                score: 0.0,
                layer: 0,
                current_zone: graph.zone_index[&context.initial_zone],
                adjusted_arrival: None,
                path_index: None,
                anchor_destinations: BTreeMap::new(),
                insertion_order: 0,
            };
            initial.upper_bound = state_upper_bound(
                destinations,
                context,
                &initial,
                parameters,
                minimum_edge_cost,
                &layer_bounds,
                relaxed_suffix.as_deref(),
            );
            let (mut best_conditioned_plan, incumbent_children) = greedy_incumbent(
                graph,
                destinations,
                context,
                &problem,
                &initial,
                parameters,
                minimum_edge_cost,
                &layer_bounds,
                relaxed_suffix.as_deref(),
            );
            report.incumbent_children_considered += incumbent_children;
            if best_conditioned_plan.is_some() {
                report.incumbent_contexts += 1;
            }
            let mut anchor_bounds = anchor_domain
                .iter()
                .map(|&anchor_zone| {
                    (
                        anchor_zone,
                        anchor_condition_upper_bound(
                            graph,
                            destinations,
                            context,
                            &problem,
                            anchor_id,
                            anchor_zone,
                            parameters,
                            minimum_edge_cost,
                            &layer_bounds,
                        ),
                    )
                })
                .collect::<Vec<_>>();
            anchor_bounds.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
            for (condition_index, (anchor_zone, upper_bound)) in
                anchor_bounds.iter().copied().enumerate()
            {
                if best_conditioned_plan
                    .as_ref()
                    .is_some_and(|best| upper_bound <= best.score)
                {
                    report.anchor_conditions_pruned +=
                        (anchor_bounds.len() - condition_index) as u64;
                    break;
                }
                report.anchor_conditions_considered += 1;
                let mut steps = context.steps.clone();
                for step in &mut steps {
                    if step.anchor_id == Some(anchor_id) {
                        step.anchor_id = None;
                        step.fixed_destination = Some(graph.zone_ids[anchor_zone]);
                    }
                }
                let conditioned_context = Context {
                    context_id: context.context_id,
                    initial_zone: context.initial_zone,
                    steps,
                };
                let conditioned_parameters = Parameters {
                    skip_infeasible: true,
                    ..parameters
                };
                let (table, conditioned_report) = search_reference_top_k_sequential(
                    graph,
                    destinations,
                    std::slice::from_ref(&conditioned_context),
                    conditioned_parameters,
                    1,
                    max_states,
                )?;
                report.add_subsearch(&conditioned_report);
                if table.destination.is_empty() {
                    continue;
                }
                let capacity = destinations.activity(anchor_step.activity_id).unwrap()[anchor_zone]
                    .unwrap()
                    .log_opportunity_capacity;
                let mut plan =
                    ranked_plans_from_output(graph, &table, conditioned_context.steps.len())
                        .pop()
                        .unwrap();
                plan.score += capacity;
                if best_conditioned_plan
                    .as_ref()
                    .is_none_or(|best| plan.score > best.score)
                {
                    best_conditioned_plan = Some(plan);
                }
            }
            if let Some(plan) = best_conditioned_plan {
                report.conditioned_anchor_contexts += 1;
                append_ranked_plans(
                    &mut output,
                    graph,
                    destinations,
                    context,
                    &problem,
                    vec![plan.zones],
                    parameters,
                )?;
                continue;
            }
            if parameters.skip_infeasible {
                continue;
            }
            return Err(SamplerError::NoFeasibleSequence {
                context_id: context.context_id,
                origin: context.initial_zone,
            });
        }

        let initial = HeapState {
            upper_bound: f64::INFINITY,
            score: 0.0,
            layer: 0,
            current_zone: graph.zone_index[&context.initial_zone],
            adjusted_arrival: None,
            path_index: None,
            anchor_destinations: BTreeMap::new(),
            insertion_order: 0,
        };
        let mut heap = BinaryHeap::new();
        let mut path: Vec<PathNode> = Vec::new();
        let mut initial = initial;
        initial.upper_bound = state_upper_bound(
            destinations,
            context,
            &initial,
            parameters,
            minimum_edge_cost,
            &layer_bounds,
            relaxed_suffix.as_deref(),
        );
        let (mut incumbent, incumbent_children) = if k == 1 {
            greedy_incumbent(
                graph,
                destinations,
                context,
                &problem,
                &initial,
                parameters,
                minimum_edge_cost,
                &layer_bounds,
                relaxed_suffix.as_deref(),
            )
        } else {
            (None, 0)
        };
        if incumbent.is_some() {
            report.incumbent_contexts += 1;
        }
        report.incumbent_children_considered += incumbent_children;
        heap.push(SearchEntry::State(initial));
        report.states_pushed += 1;
        let mut context_states_pushed = 1usize;
        let mut pending_children_in_heap = 0usize;
        let mut insertion_order = 1u64;
        let mut complete: Vec<Vec<usize>> = Vec::with_capacity(k);

        while let Some(entry) = heap.pop() {
            report.queue_entries_popped += 1;
            if incumbent
                .as_ref()
                .is_some_and(|best| entry.upper_bound() <= best.score)
            {
                break;
            }
            let state = match entry {
                SearchEntry::State(state) => state,
                SearchEntry::Siblings { mut siblings, .. } => {
                    report.sibling_entries_popped += 1;
                    let pending = siblings.children.pop().unwrap();
                    pending_children_in_heap -= 1;
                    if context_states_pushed >= max_states {
                        return Err(SamplerError::InvalidInput(format!(
                            "heap reference search exceeded max_states={max_states} in context {}",
                            context.context_id
                        )));
                    }
                    let child_order = insertion_order;
                    insertion_order += 1;
                    let child = materialize_child(
                        context,
                        &siblings.parent,
                        pending,
                        &mut path,
                        child_order,
                    );
                    if !siblings.children.is_empty() {
                        let upper_bound = siblings.children.last().unwrap().upper_bound;
                        let sibling_order = insertion_order;
                        insertion_order += 1;
                        heap.push(SearchEntry::Siblings {
                            upper_bound,
                            insertion_order: sibling_order,
                            siblings,
                        });
                    }
                    report.states_pushed += 1;
                    context_states_pushed += 1;
                    child
                }
            };
            report.states_popped += 1;
            if state.layer == context.steps.len() {
                report.complete_plans += 1;
                let mut zones = Vec::with_capacity(context.steps.len());
                let mut node_index = state.path_index;
                while let Some(index) = node_index {
                    let node = &path[index];
                    zones.push(node.zone);
                    node_index = node.parent;
                }
                zones.reverse();
                if k == 1 {
                    incumbent = Some(RankedPlan {
                        zones,
                        score: state.score,
                    });
                    break;
                }
                complete.push(zones);
                if complete.len() == k {
                    break;
                }
                continue;
            }
            if context_states_pushed >= max_states {
                return Err(SamplerError::InvalidInput(format!(
                    "heap reference search exceeded max_states={max_states} in context {}",
                    context.context_id
                )));
            }

            let (mut pending_children, children_considered) = pending_children_for_state(
                graph,
                destinations,
                context,
                &problem,
                &state,
                parameters,
                minimum_edge_cost,
                &layer_bounds,
                relaxed_suffix.as_deref(),
            );
            report.children_considered += children_considered;
            if let Some(best) = &incumbent {
                let before = pending_children.len();
                pending_children.retain(|child| child.upper_bound > best.score);
                report.children_pruned_by_incumbent += (before - pending_children.len()) as u64;
            }
            pending_children.sort_unstable_by(|left, right| {
                left.upper_bound
                    .total_cmp(&right.upper_bound)
                    .then_with(|| left.destination.cmp(&right.destination))
            });
            if let Some(pending) = pending_children.pop() {
                if !pending_children.is_empty() {
                    let pending_count = pending_children.len();
                    if context_states_pushed
                        .saturating_add(pending_children_in_heap)
                        .saturating_add(pending_count)
                        >= max_states
                    {
                        return Err(SamplerError::InvalidInput(format!(
                            "heap reference search exceeded max_states={max_states} in context {}",
                            context.context_id
                        )));
                    }
                    let upper_bound = pending_children.last().unwrap().upper_bound;
                    let sibling_order = insertion_order;
                    insertion_order += 1;
                    heap.push(SearchEntry::Siblings {
                        upper_bound,
                        insertion_order: sibling_order,
                        siblings: SiblingState {
                            parent: state.clone(),
                            children: pending_children,
                        },
                    });
                    pending_children_in_heap += pending_count;
                }
                if context_states_pushed >= max_states {
                    return Err(SamplerError::InvalidInput(format!(
                        "heap reference search exceeded max_states={max_states} in context {}",
                        context.context_id
                    )));
                }
                let child_order = insertion_order;
                insertion_order += 1;
                heap.push(SearchEntry::State(materialize_child(
                    context,
                    &state,
                    pending,
                    &mut path,
                    child_order,
                )));
                report.states_pushed += 1;
                context_states_pushed += 1;
            }
            report.maximum_heap_size = report.maximum_heap_size.max(heap.len() as u64);
        }

        let plans = if k == 1 {
            incumbent.map(|plan| vec![plan.zones]).unwrap_or_default()
        } else {
            complete
        };
        if plans.len() < k {
            if parameters.skip_infeasible {
                continue;
            }
            return Err(SamplerError::NoFeasibleSequence {
                context_id: context.context_id,
                origin: context.initial_zone,
            });
        }
        append_ranked_plans(
            &mut output,
            graph,
            destinations,
            context,
            &problem,
            plans,
            parameters,
        )?;
    }
    Ok((output, report))
}

pub fn search_reference_top_k(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    contexts: &[Context],
    parameters: Parameters,
    k: usize,
    max_states: usize,
    n_threads: Option<usize>,
) -> Result<(OutputTable, HeapSearchReport), SamplerError> {
    let compute = || {
        contexts
            .par_iter()
            .map(|context| {
                search_reference_top_k_sequential(
                    graph,
                    destinations,
                    std::slice::from_ref(context),
                    parameters,
                    k,
                    max_states,
                )
            })
            .collect::<Result<Vec<_>, SamplerError>>()
    };
    let tables = if let Some(n_threads) = n_threads {
        if n_threads == 0 {
            return Err(SamplerError::InvalidInput(
                "n_threads must be positive".to_string(),
            ));
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .map_err(|error| SamplerError::InvalidInput(error.to_string()))?
            .install(compute)?
    } else {
        compute()?
    };

    let mut output = OutputTable::default();
    let mut report = HeapSearchReport::default();
    for (context_output, context_report) in tables {
        output.extend(context_output);
        report.add_context_result(&context_report);
    }
    Ok((output, report))
}
