use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

use rayon::prelude::*;

use crate::common::{alternative_gumbel, fixed_destination_value, logaddexp, Parameters};
use crate::errors::SamplerError;
use crate::input::{Context, Step};
use crate::model::{DestinationIndex, DestinationValue, Edge, OdGraph};
use crate::output::{OutputRow, OutputTable};

const MIN_ACTIVITY_DURATION_HOURS: f64 = 1e-3;

pub(crate) struct ReferenceProblem<'a> {
    variable_by_layer: Vec<Option<usize>>,
    first_choice_by_layer: Vec<bool>,
    domains: Vec<&'a [usize]>,
    has_cross_home_anchor: bool,
}

#[derive(Clone)]
struct SegmentConfiguration {
    zones: Vec<usize>,
    log_weight: f64,
}

struct SegmentState {
    first_adjusted_departure: f64,
    end_adjusted_arrival: f64,
    log_weight: f64,
    configurations: Vec<SegmentConfiguration>,
}

struct Segment {
    ending_home_step: Step,
    ending_home_value: DestinationValue,
    states: Vec<SegmentState>,
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

pub(crate) fn build_problem<'a>(
    graph: &OdGraph,
    destinations: &'a DestinationIndex,
    context: &Context,
) -> Result<(ReferenceProblem<'a>, Vec<(usize, usize)>), SamplerError> {
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
    let mut first_choice_by_layer = Vec::with_capacity(context.steps.len());

    for (layer, step) in context.steps.iter().enumerate() {
        if step.fixed_destination.is_some() {
            variable_by_layer.push(None);
            first_choice_by_layer.push(false);
            continue;
        }
        let (variable, first_choice) = if let Some(anchor_id) = step.anchor_id {
            if let Some(&variable) = anchor_variables.get(&anchor_id) {
                if variable_activities[variable] != step.activity_id {
                    return Err(SamplerError::InvalidInput(format!(
                        "context {} uses anchor id {} for several activities",
                        context.context_id, anchor_id
                    )));
                }
                if anchor_segment_by_id[&anchor_id] != segment_by_layer[layer] {
                    has_cross_home_anchor = true;
                }
                (variable, false)
            } else {
                let variable = variable_activities.len();
                variable_activities.push(step.activity_id);
                anchor_variables.insert(anchor_id, variable);
                anchor_segment_by_id.insert(anchor_id, segment_by_layer[layer]);
                (variable, true)
            }
        } else {
            let variable = variable_activities.len();
            variable_activities.push(step.activity_id);
            (variable, true)
        };
        variable_by_layer.push(Some(variable));
        first_choice_by_layer.push(first_choice);
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
        ReferenceProblem {
            variable_by_layer,
            first_choice_by_layer,
            domains,
            has_cross_home_anchor,
        },
        ranges,
    ))
}

fn for_each_assignment(
    domains: &[&[usize]],
    assignment: &mut [usize],
    variable: usize,
    visit: &mut impl FnMut(&[usize]),
) {
    if variable == domains.len() {
        visit(assignment);
        return;
    }
    for &destination in domains[variable] {
        assignment[variable] = destination;
        for_each_assignment(domains, assignment, variable + 1, visit);
    }
}

pub(crate) fn adjusted_times(step: Step, edge: Edge) -> Option<(f64, f64)> {
    let arrival = step.arrival_time?;
    let arrival_rigidity = step.arrival_time_rigidity?;
    let departure_rigidity = step.departure_time_rigidity?;
    let reference_travel_time = (arrival - step.departure_time).clamp(0.0, 24.0);
    let delta = edge.time - reference_travel_time;
    let total_rigidity = arrival_rigidity + departure_rigidity;
    let departure_shift_share = if total_rigidity > 0.0 {
        arrival_rigidity / total_rigidity
    } else {
        0.5
    };
    let arrival_shift_share = if total_rigidity > 0.0 {
        departure_rigidity / total_rigidity
    } else {
        0.5
    };
    Some((
        step.departure_time - departure_shift_share * delta,
        arrival + arrival_shift_share * delta,
    ))
}

pub(crate) fn activity_log_weight(
    step: Step,
    edge: Edge,
    destination_value: DestinationValue,
    duration: f64,
    attraction: f64,
    parameters: Parameters,
) -> f64 {
    let activity_coefficient = if parameters.use_shadow_prices {
        destination_value.country_value_coefficient * step.value_of_time
            + destination_value.shadow_price
    } else {
        destination_value.country_value_coefficient
            * destination_value.saturation_utility
            * step.value_of_time
    };
    let activity_utility = activity_coefficient
        * step.mean_duration_per_person
        * (duration / step.min_activity_time).ln().max(0.0);
    attraction + parameters.logit_scale * (activity_utility - edge.cost)
}

/// Score one activity once its incoming and outgoing zones are known.
///
/// An ordinary activity factor is ternary: its incoming edge determines the
/// adjusted arrival and travel cost, while its outgoing edge determines the
/// next adjusted departure and therefore the activity duration. The terminal
/// fixed home has no outgoing edge in the observed day.
pub(crate) fn score_local_weight(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    context: &Context,
    problem: &ReferenceProblem<'_>,
    layer: usize,
    origin: usize,
    destination: usize,
    next_destination: Option<usize>,
    parameters: Parameters,
) -> Option<f64> {
    let step = context.steps[layer];
    let edge = graph.edge_to(origin, destination)?;
    let (_, arrival) = adjusted_times(step, edge)?;
    let terminal_fixed_home = layer + 1 == context.steps.len() && step.fixed_destination.is_some();
    let duration = if terminal_fixed_home {
        MIN_ACTIVITY_DURATION_HOURS
    } else if parameters.update_plan_timings {
        let next_step = context.steps.get(layer + 1)?;
        let next_edge = graph.edge_to(destination, next_destination?)?;
        let (next_departure, _) = adjusted_times(*next_step, next_edge)?;
        let duration = next_departure - arrival;
        if duration <= 0.0 {
            return None;
        }
        duration
    } else {
        step.duration_per_person
    };
    let destination_value =
        fixed_destination_value(destinations.activity(step.activity_id), destination);
    let attraction = if step.fixed_destination.is_some() || !problem.first_choice_by_layer[layer] {
        0.0
    } else if destination_value.log_opportunity_capacity.is_finite() {
        destination_value.log_opportunity_capacity
    } else {
        return None;
    };
    Some(activity_log_weight(
        step,
        edge,
        destination_value,
        duration,
        attraction,
        parameters,
    ))
}

pub(crate) fn score_zones(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    context: &Context,
    problem: &ReferenceProblem<'_>,
    zones: &[usize],
    parameters: Parameters,
) -> Option<(f64, Vec<f64>)> {
    let initial_zone = *graph.zone_index.get(&context.initial_zone)?;
    let mut local_weights = Vec::with_capacity(zones.len());
    let mut total = 0.0;
    for layer in 0..context.steps.len() {
        let origin = if layer == 0 {
            initial_zone
        } else {
            *zones.get(layer - 1)?
        };
        let local = score_local_weight(
            graph,
            destinations,
            context,
            problem,
            layer,
            origin,
            *zones.get(layer)?,
            zones.get(layer + 1).copied(),
            parameters,
        )?;
        total += local;
        local_weights.push(local);
    }
    Some((total, local_weights))
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
    problem: &ReferenceProblem<'_>,
    plans: Vec<Vec<usize>>,
    parameters: Parameters,
) -> Result<(), SamplerError> {
    let mut scored = Vec::with_capacity(plans.len());
    for zones in plans {
        let Some((score, local_weights)) =
            score_zones(graph, destinations, context, problem, &zones, parameters)
        else {
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
    problem: &ReferenceProblem<'_>,
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
    problem: &ReferenceProblem<'_>,
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
    problem: &ReferenceProblem<'_>,
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
                let attraction =
                    if step.fixed_destination.is_some() || !problem.first_choice_by_layer[layer] {
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
    problem: &ReferenceProblem<'_>,
    layer: usize,
) -> f64 {
    let step = context.steps[layer];
    if step.fixed_destination.is_some() || !problem.first_choice_by_layer[layer] {
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
    problem: &ReferenceProblem<'_>,
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
        bound += if is_conditioned_anchor && problem.first_choice_by_layer[layer] {
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
    problem: &'a ReferenceProblem<'a>,
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
    problem: &ReferenceProblem<'_>,
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
                (adjusted_departure - state.adjusted_arrival.unwrap())
                    .max(MIN_ACTIVITY_DURATION_HOURS)
            } else {
                context.steps[layer - 1].duration_per_person
            };
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
            if step.fixed_destination.is_some() || !problem.first_choice_by_layer[layer] {
                0.0
            } else {
                destination_value.log_opportunity_capacity
            };
        child.score += attraction - parameters.logit_scale * edge.cost;

        if layer + 1 == context.steps.len() {
            let duration = if parameters.update_plan_timings {
                (step.next_departure_time - adjusted_arrival).max(MIN_ACTIVITY_DURATION_HOURS)
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
    problem: &ReferenceProblem<'_>,
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
        let (problem, ranges) = build_problem(graph, destinations, context)?;
        let lattice = problem.domains.iter().fold(1u128, |count, domain| {
            count.saturating_mul(domain.len() as u128)
        });
        report.assignment_lattice = report.assignment_lattice.saturating_add(lattice);

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
                        build_problem(graph, destinations, &segment_context)?;
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
                    report.conditioned_anchor_contexts +=
                        segment_report.conditioned_anchor_contexts;
                    report.anchor_conditions_considered +=
                        segment_report.anchor_conditions_considered;
                    report.anchor_conditions_pruned += segment_report.anchor_conditions_pruned;
                    report.incumbent_contexts += segment_report.incumbent_contexts;
                    report.incumbent_children_considered +=
                        segment_report.incumbent_children_considered;
                    report.children_pruned_by_incumbent +=
                        segment_report.children_pruned_by_incumbent;
                    report.queue_entries_popped += segment_report.queue_entries_popped;
                    report.sibling_entries_popped += segment_report.sibling_entries_popped;
                    report.states_popped += segment_report.states_popped;
                    report.states_pushed += segment_report.states_pushed;
                    report.children_considered += segment_report.children_considered;
                    report.complete_plans += segment_report.complete_plans;
                    report.maximum_heap_size = report
                        .maximum_heap_size
                        .max(segment_report.maximum_heap_size);
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
                report.split_contexts += conditioned_report.split_contexts;
                report.conditioned_anchor_contexts +=
                    conditioned_report.conditioned_anchor_contexts;
                report.anchor_conditions_considered +=
                    conditioned_report.anchor_conditions_considered;
                report.anchor_conditions_pruned += conditioned_report.anchor_conditions_pruned;
                report.incumbent_contexts += conditioned_report.incumbent_contexts;
                report.incumbent_children_considered +=
                    conditioned_report.incumbent_children_considered;
                report.children_pruned_by_incumbent +=
                    conditioned_report.children_pruned_by_incumbent;
                report.queue_entries_popped += conditioned_report.queue_entries_popped;
                report.sibling_entries_popped += conditioned_report.sibling_entries_popped;
                report.states_popped += conditioned_report.states_popped;
                report.states_pushed += conditioned_report.states_pushed;
                report.children_considered += conditioned_report.children_considered;
                report.complete_plans += conditioned_report.complete_plans;
                report.maximum_heap_size = report
                    .maximum_heap_size
                    .max(conditioned_report.maximum_heap_size);
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
        report.contexts += context_report.contexts;
        report.split_contexts += context_report.split_contexts;
        report.conditioned_anchor_contexts += context_report.conditioned_anchor_contexts;
        report.anchor_conditions_considered += context_report.anchor_conditions_considered;
        report.anchor_conditions_pruned += context_report.anchor_conditions_pruned;
        report.incumbent_contexts += context_report.incumbent_contexts;
        report.incumbent_children_considered += context_report.incumbent_children_considered;
        report.children_pruned_by_incumbent += context_report.children_pruned_by_incumbent;
        report.queue_entries_popped += context_report.queue_entries_popped;
        report.sibling_entries_popped += context_report.sibling_entries_popped;
        report.states_popped += context_report.states_popped;
        report.states_pushed += context_report.states_pushed;
        report.children_considered += context_report.children_considered;
        report.complete_plans += context_report.complete_plans;
        report.maximum_heap_size = report
            .maximum_heap_size
            .max(context_report.maximum_heap_size);
        report.assignment_lattice = report
            .assignment_lattice
            .saturating_add(context_report.assignment_lattice);
    }
    Ok((output, report))
}

fn zones_from_assignment(
    graph: &OdGraph,
    context: &Context,
    problem: &ReferenceProblem<'_>,
    assignment: &[usize],
) -> Option<Vec<usize>> {
    context
        .steps
        .iter()
        .enumerate()
        .map(|(layer, step)| {
            if let Some(zone_id) = step.fixed_destination {
                graph.zone_index.get(&zone_id).copied()
            } else {
                Some(assignment[problem.variable_by_layer[layer]?])
            }
        })
        .collect()
}

fn sample_whole_context(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    context: &Context,
    problem: &ReferenceProblem<'_>,
    parameters: Parameters,
    max_assignments: usize,
) -> Result<Vec<Vec<usize>>, SamplerError> {
    let assignment_count = problem
        .domains
        .iter()
        .try_fold(1usize, |count, domain| count.checked_mul(domain.len()));
    if assignment_count.is_none_or(|count| count > max_assignments) {
        return Err(SamplerError::InvalidInput(format!(
            "context {} has more than {} complete destination assignments; the ternary reference sampler is limited to small domains",
            context.context_id, max_assignments
        )));
    }

    let mut assignment = vec![0usize; problem.domains.len()];
    let mut best_scores = vec![f64::NEG_INFINITY; parameters.n_draws as usize];
    let mut selected = vec![None; parameters.n_draws as usize];
    let mut alternative_index = 0u64;
    for_each_assignment(&problem.domains, &mut assignment, 0, &mut |assignment| {
        let Some(zones) = zones_from_assignment(graph, context, problem, assignment) else {
            alternative_index += 1;
            return;
        };
        let Some((score, _)) =
            score_zones(graph, destinations, context, problem, &zones, parameters)
        else {
            alternative_index += 1;
            return;
        };
        for draw_id in 1..=parameters.n_draws {
            let perturbed = score
                + alternative_gumbel(
                    parameters.seed,
                    context.context_id,
                    draw_id,
                    alternative_index,
                );
            let draw_index = (draw_id - 1) as usize;
            if perturbed > best_scores[draw_index] {
                best_scores[draw_index] = perturbed;
                selected[draw_index] = Some(zones.clone());
            }
        }
        alternative_index += 1;
    });
    selected
        .into_iter()
        .map(|zones| {
            zones.ok_or(SamplerError::NoFeasibleSequence {
                context_id: context.context_id,
                origin: context.initial_zone,
            })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn score_segment_configuration(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    context: &Context,
    problem: &ReferenceProblem<'_>,
    start: usize,
    end: usize,
    zones: Vec<usize>,
    is_last_segment: bool,
    parameters: Parameters,
) -> Option<(SegmentConfiguration, f64, f64, usize, usize)> {
    let home = *graph.zone_index.get(&context.initial_zone)?;
    let mut origin = home;
    let mut edges = Vec::with_capacity(zones.len());
    let mut departures = Vec::with_capacity(zones.len());
    let mut arrivals = Vec::with_capacity(zones.len());
    for (&destination, &step) in zones.iter().zip(&context.steps[start..end]) {
        let edge = graph.edge_to(origin, destination)?;
        let (departure, arrival) = adjusted_times(step, edge)?;
        edges.push(edge);
        departures.push(departure);
        arrivals.push(arrival);
        origin = destination;
    }

    let mut log_weight = 0.0;
    for local_layer in 0..zones.len() {
        let layer = start + local_layer;
        let step = context.steps[layer];
        let destination = zones[local_layer];
        let destination_value =
            fixed_destination_value(destinations.activity(step.activity_id), destination);
        let defer_home_activity =
            !is_last_segment && local_layer + 1 == zones.len() && destination == home;
        let duration = if defer_home_activity {
            0.0
        } else if parameters.update_plan_timings {
            let next_departure = departures
                .get(local_layer + 1)
                .copied()
                .unwrap_or(step.next_departure_time);
            (next_departure - arrivals[local_layer]).max(MIN_ACTIVITY_DURATION_HOURS)
        } else {
            step.duration_per_person
        };
        let attraction =
            if step.fixed_destination.is_some() || !problem.first_choice_by_layer[layer] {
                0.0
            } else if destination_value.log_opportunity_capacity.is_finite() {
                destination_value.log_opportunity_capacity
            } else {
                return None;
            };
        if defer_home_activity && parameters.update_plan_timings {
            log_weight += attraction - parameters.logit_scale * edges[local_layer].cost;
        } else {
            log_weight += activity_log_weight(
                step,
                edges[local_layer],
                destination_value,
                duration,
                attraction,
                parameters,
            );
        }
    }

    let first_zone = zones.first().copied().unwrap_or(home);
    let end_origin = if zones.last().copied() == Some(home) && zones.len() >= 2 {
        zones[zones.len() - 2]
    } else {
        zones.last().copied().unwrap_or(home)
    };
    Some((
        SegmentConfiguration { zones, log_weight },
        departures.first().copied().unwrap_or(0.0),
        arrivals.last().copied().unwrap_or(0.0),
        first_zone,
        end_origin,
    ))
}

fn build_segments(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    context: &Context,
    problem: &ReferenceProblem<'_>,
    ranges: &[(usize, usize)],
    parameters: Parameters,
    max_assignments: usize,
) -> Result<Vec<Segment>, SamplerError> {
    let mut segments = Vec::with_capacity(ranges.len());
    let mut total_configurations = 0usize;
    for (segment_index, &(start, end)) in ranges.iter().enumerate() {
        let variables = problem.variable_by_layer[start..end]
            .iter()
            .flatten()
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let domains = variables
            .iter()
            .map(|&variable| problem.domains[variable])
            .collect::<Vec<_>>();
        let assignment_count = domains
            .iter()
            .try_fold(1usize, |count, domain| count.checked_mul(domain.len()));
        total_configurations = assignment_count
            .and_then(|count| total_configurations.checked_add(count))
            .filter(|&count| count <= max_assignments)
            .ok_or_else(|| {
                SamplerError::InvalidInput(format!(
                    "context {} needs more than {} enumerated home-tour configurations; increase max_assignments only for a controlled reference run",
                    context.context_id, max_assignments
                ))
            })?;

        let variable_positions = variables
            .iter()
            .enumerate()
            .map(|(position, &variable)| (variable, position))
            .collect::<BTreeMap<_, _>>();
        let mut grouped: BTreeMap<(usize, usize), Vec<SegmentConfiguration>> = BTreeMap::new();
        let mut boundary_values = BTreeMap::new();
        let mut assignment = vec![0usize; domains.len()];
        for_each_assignment(&domains, &mut assignment, 0, &mut |assignment| {
            let zones = context.steps[start..end]
                .iter()
                .enumerate()
                .map(|(offset, step)| {
                    if let Some(zone_id) = step.fixed_destination {
                        graph.zone_index.get(&zone_id).copied()
                    } else {
                        let variable = problem.variable_by_layer[start + offset]?;
                        Some(assignment[variable_positions[&variable]])
                    }
                })
                .collect::<Option<Vec<_>>>();
            let Some(zones) = zones else {
                return;
            };
            let Some((configuration, first_departure, end_arrival, first_zone, end_origin)) =
                score_segment_configuration(
                    graph,
                    destinations,
                    context,
                    problem,
                    start,
                    end,
                    zones,
                    segment_index + 1 == ranges.len(),
                    parameters,
                )
            else {
                return;
            };
            grouped
                .entry((first_zone, end_origin))
                .or_default()
                .push(configuration);
            boundary_values.insert((first_zone, end_origin), (first_departure, end_arrival));
        });
        let ending_home_step = context.steps[end - 1];
        let states = grouped
            .into_iter()
            .map(|((first_zone, end_origin), configurations)| {
                let log_weight = configurations
                    .iter()
                    .fold(f64::NEG_INFINITY, |total, configuration| {
                        logaddexp(total, configuration.log_weight)
                    });
                let (first_adjusted_departure, end_adjusted_arrival) =
                    boundary_values[&(first_zone, end_origin)];
                SegmentState {
                    first_adjusted_departure,
                    end_adjusted_arrival,
                    log_weight,
                    configurations,
                }
            })
            .collect::<Vec<_>>();
        if states.is_empty() {
            return Err(SamplerError::NoFeasibleSequence {
                context_id: context.context_id,
                origin: context.initial_zone,
            });
        }
        let ending_home_destination = ending_home_step
            .fixed_destination
            .and_then(|zone_id| graph.zone_index.get(&zone_id).copied())
            .expect("home-ending segments have a fixed destination");
        segments.push(Segment {
            ending_home_step,
            ending_home_value: fixed_destination_value(
                destinations.activity(ending_home_step.activity_id),
                ending_home_destination,
            ),
            states,
        });
    }
    Ok(segments)
}

fn boundary_log_weight(
    previous: &Segment,
    previous_state: &SegmentState,
    next_state: &SegmentState,
    parameters: Parameters,
) -> f64 {
    if !parameters.update_plan_timings {
        return 0.0;
    }
    let step = previous.ending_home_step;
    let duration = (next_state.first_adjusted_departure - previous_state.end_adjusted_arrival)
        .max(MIN_ACTIVITY_DURATION_HOURS);
    activity_log_weight(
        step,
        Edge {
            destination: 0,
            cost: 0.0,
            time: 0.0,
        },
        previous.ending_home_value,
        duration,
        0.0,
        parameters,
    )
}

fn choose_index(
    scores: impl Iterator<Item = (usize, f64)>,
    parameters: Parameters,
    context_id: u64,
    draw_id: u32,
    stage: u64,
) -> Option<usize> {
    scores
        .filter(|(_, score)| score.is_finite())
        .map(|(index, score)| {
            (
                index,
                score
                    + alternative_gumbel(
                        parameters.seed ^ stage.wrapping_mul(0x9E3779B97F4A7C15),
                        context_id,
                        draw_id,
                        index as u64,
                    ),
            )
        })
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .map(|(index, _)| index)
}

fn sample_segmented_context(
    context: &Context,
    segments: &[Segment],
    parameters: Parameters,
) -> Result<Vec<Vec<usize>>, SamplerError> {
    let mut alpha = Vec::with_capacity(segments.len());
    alpha.push(
        segments[0]
            .states
            .iter()
            .map(|state| state.log_weight)
            .collect::<Vec<_>>(),
    );
    for segment_index in 1..segments.len() {
        let previous = &segments[segment_index - 1];
        let current = &segments[segment_index];
        let values = current
            .states
            .iter()
            .map(|current_state| {
                let incoming = previous.states.iter().enumerate().fold(
                    f64::NEG_INFINITY,
                    |total, (previous_index, previous_state)| {
                        logaddexp(
                            total,
                            alpha[segment_index - 1][previous_index]
                                + boundary_log_weight(
                                    previous,
                                    previous_state,
                                    current_state,
                                    parameters,
                                ),
                        )
                    },
                );
                current_state.log_weight + incoming
            })
            .collect::<Vec<_>>();
        alpha.push(values);
    }

    let mut results = Vec::with_capacity(parameters.n_draws as usize);
    for draw_id in 1..=parameters.n_draws {
        let last_segment = segments.len() - 1;
        let mut selected_states = vec![0usize; segments.len()];
        selected_states[last_segment] = choose_index(
            alpha[last_segment].iter().copied().enumerate(),
            parameters,
            context.context_id,
            draw_id,
            10_000 + last_segment as u64,
        )
        .ok_or(SamplerError::NoFeasibleSequence {
            context_id: context.context_id,
            origin: context.initial_zone,
        })?;
        for segment_index in (1..segments.len()).rev() {
            let current_state = &segments[segment_index].states[selected_states[segment_index]];
            selected_states[segment_index - 1] = choose_index(
                segments[segment_index - 1].states.iter().enumerate().map(
                    |(previous_index, previous_state)| {
                        (
                            previous_index,
                            alpha[segment_index - 1][previous_index]
                                + boundary_log_weight(
                                    &segments[segment_index - 1],
                                    previous_state,
                                    current_state,
                                    parameters,
                                ),
                        )
                    },
                ),
                parameters,
                context.context_id,
                draw_id,
                20_000 + segment_index as u64,
            )
            .expect("selected continuation has a feasible predecessor");
        }

        let mut zones = Vec::with_capacity(context.steps.len());
        for (segment_index, &state_index) in selected_states.iter().enumerate() {
            let state = &segments[segment_index].states[state_index];
            let configuration_index = choose_index(
                state
                    .configurations
                    .iter()
                    .enumerate()
                    .map(|(index, configuration)| (index, configuration.log_weight)),
                parameters,
                context.context_id,
                draw_id,
                30_000 + segment_index as u64,
            )
            .expect("selected segment state contains a feasible configuration");
            zones.extend_from_slice(&state.configurations[configuration_index].zones);
        }
        results.push(zones);
    }
    Ok(results)
}

fn append_results(
    output: &mut OutputTable,
    graph: &OdGraph,
    destinations: &DestinationIndex,
    context: &Context,
    problem: &ReferenceProblem<'_>,
    selected_zones: Vec<Vec<usize>>,
    parameters: Parameters,
) {
    for (draw_index, zones) in selected_zones.into_iter().enumerate() {
        let (_, local_weights) =
            score_zones(graph, destinations, context, problem, &zones, parameters)
                .expect("selected reference assignments are feasible");
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
                draw_id: draw_index as u32 + 1,
                layer: layer as u32,
                origin: graph.zone_ids[origin],
                destination: graph.zone_ids[destination],
                local_log_weight: local_weights[layer],
                total_log_weight: suffix_values[layer],
            });
            origin = destination;
        }
    }
}

pub fn sample_reference_all(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    contexts: &[Context],
    parameters: Parameters,
    max_assignments: usize,
) -> Result<OutputTable, SamplerError> {
    let mut output = OutputTable::default();
    for context in contexts {
        let result = (|| -> Result<(), SamplerError> {
            let (problem, ranges) = build_problem(graph, destinations, context)?;
            let selected = if ranges.len() > 1 && !problem.has_cross_home_anchor {
                let segments = build_segments(
                    graph,
                    destinations,
                    context,
                    &problem,
                    &ranges,
                    parameters,
                    max_assignments,
                )?;
                sample_segmented_context(context, &segments, parameters)?
            } else {
                sample_whole_context(
                    graph,
                    destinations,
                    context,
                    &problem,
                    parameters,
                    max_assignments,
                )?
            };
            append_results(
                &mut output,
                graph,
                destinations,
                context,
                &problem,
                selected,
                parameters,
            );
            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(SamplerError::NoFeasibleSequence { .. }) if parameters.skip_infeasible => {}
            Err(error) => return Err(error),
        }
    }
    Ok(output)
}
