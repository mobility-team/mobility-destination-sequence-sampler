//! Shared destination-plan problem construction and exact local scoring.

use std::collections::BTreeMap;

use crate::common::{fixed_destination_value, Parameters};
use crate::errors::SamplerError;
use crate::input::{Context, Step};
use crate::model::{DestinationIndex, DestinationValue, Edge, OdGraph};

pub(crate) const MIN_ACTIVITY_DURATION_HOURS: f64 = 1e-3;

pub(crate) struct SearchProblem<'a> {
    pub(crate) variable_by_layer: Vec<Option<usize>>,
    pub(crate) first_choice_by_layer: Vec<bool>,
    pub(crate) domains: Vec<&'a [usize]>,
    pub(crate) has_cross_home_anchor: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct ScoringInputs<'a> {
    pub(crate) graph: &'a OdGraph,
    pub(crate) destinations: &'a DestinationIndex,
    pub(crate) context: &'a Context,
    pub(crate) problem: &'a SearchProblem<'a>,
    pub(crate) parameters: Parameters,
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
) -> Result<(SearchProblem<'a>, Vec<(usize, usize)>), SamplerError> {
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
        SearchProblem {
            variable_by_layer,
            first_choice_by_layer,
            domains,
            has_cross_home_anchor,
        },
        ranges,
    ))
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

fn activity_log_weight(
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

pub(crate) fn score_local_weight(
    inputs: ScoringInputs<'_>,
    layer: usize,
    origin: usize,
    destination: usize,
    next_destination: Option<usize>,
) -> Option<f64> {
    let step = inputs.context.steps[layer];
    let edge = inputs.graph.edge_to(origin, destination)?;
    let (_, arrival) = adjusted_times(step, edge)?;
    let terminal_fixed_destination =
        layer + 1 == inputs.context.steps.len() && step.fixed_destination.is_some();
    let duration = if terminal_fixed_destination {
        MIN_ACTIVITY_DURATION_HOURS
    } else if inputs.parameters.update_plan_timings {
        let next_step = inputs.context.steps.get(layer + 1)?;
        let next_edge = inputs.graph.edge_to(destination, next_destination?)?;
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
        fixed_destination_value(inputs.destinations.activity(step.activity_id), destination);
    let attraction =
        if step.fixed_destination.is_some() || !inputs.problem.first_choice_by_layer[layer] {
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
        inputs.parameters,
    ))
}

pub(crate) fn score_zones(inputs: ScoringInputs<'_>, zones: &[usize]) -> Option<(f64, Vec<f64>)> {
    let initial_zone = *inputs.graph.zone_index.get(&inputs.context.initial_zone)?;
    let mut local_weights = Vec::with_capacity(zones.len());
    let mut total = 0.0;
    for layer in 0..inputs.context.steps.len() {
        let origin = if layer == 0 {
            initial_zone
        } else {
            *zones.get(layer - 1)?
        };
        let local = score_local_weight(
            inputs,
            layer,
            origin,
            *zones.get(layer)?,
            zones.get(layer + 1).copied(),
        )?;
        total += local;
        local_weights.push(local);
    }
    Some((total, local_weights))
}
