//! Shared destination-plan problem construction and exact local scoring.

use std::collections::BTreeMap;

use crate::errors::SamplerError;
use crate::input::{Context, Step};
use crate::model::{DestinationIndex, DestinationValue, Edge, OdGraph};

pub(crate) const MIN_ACTIVITY_DURATION_HOURS: f64 = 1e-3;

#[derive(Clone, Copy, Debug)]
pub struct Parameters {
    pub logit_scale: f64,
    pub update_plan_timings: bool,
    pub use_shadow_prices: bool,
    pub skip_infeasible: bool,
}

#[inline]
pub(crate) fn fixed_destination_value(
    activity_values: Option<&[Option<DestinationValue>]>,
    destination: usize,
) -> DestinationValue {
    activity_values
        .and_then(|values| values[destination])
        .unwrap_or(DestinationValue {
            log_opportunity_capacity: 0.0,
            country_value_coefficient: 1.0,
            saturation_utility: 1.0,
            shadow_price: 0.0,
        })
}

pub(crate) struct ScoringProblem {
    first_choice_by_layer: Vec<bool>,
}

impl ScoringProblem {
    #[inline]
    pub(crate) fn is_first_choice(&self, layer: usize) -> bool {
        self.first_choice_by_layer[layer]
    }
}

pub(crate) fn build_scoring_problem(context: &Context) -> Result<ScoringProblem, SamplerError> {
    let mut anchor_activities = BTreeMap::new();
    let mut first_choice_by_layer = Vec::with_capacity(context.steps.len());
    for step in &context.steps {
        if step.fixed_destination.is_some() {
            first_choice_by_layer.push(false);
            continue;
        }
        let first_choice = if let Some(anchor_id) = step.anchor_id {
            if let Some(activity_id) = anchor_activities.get(&anchor_id) {
                if *activity_id != step.activity_id {
                    return Err(SamplerError::InvalidInput(format!(
                        "context {} uses anchor id {} for several activities",
                        context.context_id, anchor_id
                    )));
                }
                false
            } else {
                anchor_activities.insert(anchor_id, step.activity_id);
                true
            }
        } else {
            true
        };
        first_choice_by_layer.push(first_choice);
    }
    Ok(ScoringProblem {
        first_choice_by_layer,
    })
}

#[derive(Clone, Copy)]
pub(crate) struct ScoringInputs<'a> {
    pub(crate) graph: &'a OdGraph,
    pub(crate) destinations: &'a DestinationIndex,
    pub(crate) context: &'a Context,
    pub(crate) problem: &'a ScoringProblem,
    pub(crate) parameters: Parameters,
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

#[derive(Clone, Copy)]
pub(crate) struct PreparedLocalScorer<'a> {
    step: Step,
    next_step: Option<Step>,
    activity_values: Option<&'a [Option<DestinationValue>]>,
    first_choice: bool,
    terminal_fixed_destination: bool,
    parameters: Parameters,
}

impl<'a> PreparedLocalScorer<'a> {
    pub(crate) fn new(
        destinations: &'a DestinationIndex,
        context: &Context,
        problem: &ScoringProblem,
        parameters: Parameters,
        layer: usize,
    ) -> Self {
        let step = context.steps[layer];
        Self {
            step,
            next_step: context.steps.get(layer + 1).copied(),
            activity_values: destinations.activity(step.activity_id),
            first_choice: problem.first_choice_by_layer[layer],
            terminal_fixed_destination: layer + 1 == context.steps.len()
                && step.fixed_destination.is_some(),
            parameters,
        }
    }

    #[inline]
    pub(crate) fn score_from_times(
        self,
        destination: usize,
        edge: Edge,
        arrival: f64,
        next_departure: Option<f64>,
    ) -> Option<f64> {
        let duration = if self.terminal_fixed_destination {
            MIN_ACTIVITY_DURATION_HOURS
        } else if self.parameters.update_plan_timings {
            let duration = next_departure? - arrival;
            if duration <= 0.0 {
                return None;
            }
            duration
        } else {
            self.step.duration_per_person
        };
        let destination_value = fixed_destination_value(self.activity_values, destination);
        let attraction = if self.step.fixed_destination.is_some() || !self.first_choice {
            0.0
        } else if destination_value.log_opportunity_capacity.is_finite() {
            destination_value.log_opportunity_capacity
        } else {
            return None;
        };
        Some(activity_log_weight(
            self.step,
            edge,
            destination_value,
            duration,
            attraction,
            self.parameters,
        ))
    }

    #[inline]
    pub(crate) fn score_edges(
        self,
        destination: usize,
        edge: Edge,
        next_edge: Option<Edge>,
    ) -> Option<f64> {
        let (_, arrival) = adjusted_times(self.step, edge)?;
        let next_departure =
            if self.terminal_fixed_destination || !self.parameters.update_plan_timings {
                None
            } else {
                Some(adjusted_times(self.next_step?, next_edge?)?.0)
            };
        self.score_from_times(destination, edge, arrival, next_departure)
    }
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
    let terminal_fixed_destination =
        layer + 1 == inputs.context.steps.len() && step.fixed_destination.is_some();
    let next_edge = if terminal_fixed_destination {
        None
    } else {
        Some(inputs.graph.edge_to(destination, next_destination?)?)
    };
    score_local_weight_edges(inputs, layer, destination, edge, next_edge)
}

/// Exact local scoring with already-resolved inbound and outgoing OD edges.
/// Map builders use this to reuse fixed legs without changing the factor
/// formula or its rigidity feasibility checks.
#[inline]
pub(crate) fn score_local_weight_edges(
    inputs: ScoringInputs<'_>,
    layer: usize,
    destination: usize,
    edge: Edge,
    next_edge: Option<Edge>,
) -> Option<f64> {
    let step = inputs.context.steps[layer];
    let (_, arrival) = adjusted_times(step, edge)?;
    let terminal_fixed_destination =
        layer + 1 == inputs.context.steps.len() && step.fixed_destination.is_some();
    let next_departure = if terminal_fixed_destination || !inputs.parameters.update_plan_timings {
        None
    } else {
        let next_step = inputs.context.steps.get(layer + 1)?;
        let next_edge = next_edge?;
        Some(adjusted_times(*next_step, next_edge)?.0)
    };
    score_local_weight_from_times(inputs, layer, destination, edge, arrival, next_departure)
}

/// Exact local scoring with already-adjusted timing terms.
///
/// Factor-map builders use this when an inbound arrival or next departure is
/// fixed across an entire destination map. It preserves the exact scorer's
/// rigidity and feasibility semantics while avoiding repeated adjustment of
/// that fixed leg.
#[inline]
pub(crate) fn score_local_weight_from_times(
    inputs: ScoringInputs<'_>,
    layer: usize,
    destination: usize,
    edge: Edge,
    arrival: f64,
    next_departure: Option<f64>,
) -> Option<f64> {
    let step = inputs.context.steps[layer];
    let terminal_fixed_destination =
        layer + 1 == inputs.context.steps.len() && step.fixed_destination.is_some();
    let duration = if terminal_fixed_destination {
        MIN_ACTIVITY_DURATION_HOURS
    } else if inputs.parameters.update_plan_timings {
        let duration = next_departure? - arrival;
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
