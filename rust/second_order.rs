use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

use crate::errors::SamplerError;
use crate::input::{Context, Step};
use crate::model::{DestinationIndex, DestinationValue, OdGraph};
use crate::sampler::{fixed_destination_value, logaddexp, Parameters};
use crate::ternary_reference::{activity_log_weight, adjusted_times};

// Status: paused reduced-resolution research path; not the active redesign.

#[derive(Clone, Copy, Debug, Default)]
struct FeasibilityStats {
    duration_checks: u64,
    duration_infeasible: u64,
    scored_transitions: u64,
    pair_states: u64,
    feasible_pair_states: u64,
    forward_pair_states: u64,
    forward_reachable_pair_states: u64,
    forward_time_edge_scans: u64,
    forward_time_cutoffs: u64,
    backward_time_edge_scans: u64,
    backward_time_cutoffs: u64,
    corridor_pair_states: u64,
}

impl FeasibilityStats {
    fn add(&mut self, other: Self) {
        self.duration_checks += other.duration_checks;
        self.duration_infeasible += other.duration_infeasible;
        self.scored_transitions += other.scored_transitions;
        self.pair_states += other.pair_states;
        self.feasible_pair_states += other.feasible_pair_states;
        self.forward_pair_states += other.forward_pair_states;
        self.forward_reachable_pair_states += other.forward_reachable_pair_states;
        self.forward_time_edge_scans += other.forward_time_edge_scans;
        self.forward_time_cutoffs += other.forward_time_cutoffs;
        self.backward_time_edge_scans += other.backward_time_edge_scans;
        self.backward_time_cutoffs += other.backward_time_cutoffs;
        self.corridor_pair_states += other.corridor_pair_states;
    }
}

#[inline]
fn duration_is_feasible(step: Step, duration: f64) -> bool {
    duration >= 0.0 && (step.activity_id == 0 || duration > 0.0)
}

#[derive(Debug)]
pub struct SecondOrderResult {
    pub context_ids: Vec<u64>,
    pub log_partitions: Vec<f64>,
    pub first_destination_probabilities: Vec<f64>,
    pub zone_ids: Vec<u32>,
    pub wall_time: Duration,
    pub anchor_conditions: usize,
    pub infeasible_contexts: usize,
    pub duration_checks: u64,
    pub duration_infeasible: u64,
    pub scored_transitions: u64,
    pub pair_states: u64,
    pub feasible_pair_states: u64,
    pub forward_pair_states: u64,
    pub forward_reachable_pair_states: u64,
    pub forward_time_edge_scans: u64,
    pub forward_time_cutoffs: u64,
    pub backward_time_edge_scans: u64,
    pub backward_time_cutoffs: u64,
    pub corridor_pair_states: u64,
}

struct ConditionalResult {
    log_partition: f64,
    first_probability: Vec<f64>,
    stats: FeasibilityStats,
}

struct ContextResult {
    context_id: u64,
    conditional: ConditionalResult,
    anchor_conditions: usize,
}

/// Exact prefix feasibility for each `(previous, current)` pair state.
///
/// A state at layer `i` is reachable when there is a route from the initial
/// home location through all activities before `i`, with every completed
/// activity having a feasible adjusted duration. The current activity is not
/// checked yet because its duration also needs the following destination.
struct ForwardFeasibility {
    reachable_by_layer: Vec<Vec<u8>>,
    stats: FeasibilityStats,
}

#[inline]
fn destination_value(
    destinations: &DestinationIndex,
    step: Step,
    destination: usize,
) -> DestinationValue {
    fixed_destination_value(destinations.activity(step.activity_id), destination)
}

#[inline]
fn attraction(
    destinations: &DestinationIndex,
    step: Step,
    destination: usize,
    first_choice: bool,
) -> Option<f64> {
    if step.fixed_destination.is_some() || !first_choice {
        return Some(0.0);
    }
    let value = destination_value(destinations, step, destination);
    value
        .log_opportunity_capacity
        .is_finite()
        .then_some(value.log_opportunity_capacity)
}

fn build_domains(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    context: &Context,
    anchor_id: Option<u32>,
    anchor_destination: Option<usize>,
) -> Result<Vec<Vec<usize>>, SamplerError> {
    context
        .steps
        .iter()
        .map(|step| {
            if let Some(zone_id) = step.fixed_destination {
                return graph
                    .zone_index
                    .get(&zone_id)
                    .copied()
                    .map(|zone| vec![zone])
                    .ok_or_else(|| {
                        SamplerError::InvalidInput(format!(
                            "context {} uses unknown fixed destination {}",
                            context.context_id, zone_id
                        ))
                    });
            }
            if step.anchor_id == anchor_id && anchor_id.is_some() {
                return Ok(vec![anchor_destination.expect(
                    "an anchor destination is present when an anchor is conditioned",
                )]);
            }
            destinations
                .domain(step.activity_id)
                .filter(|domain| !domain.is_empty())
                .map(|domain| domain.to_vec())
                .ok_or(SamplerError::NoFeasibleSequence {
                    context_id: context.context_id,
                    origin: context.initial_zone,
                })
        })
        .collect()
}

fn first_choice_by_layer(context: &Context) -> Vec<bool> {
    let mut seen_anchors = BTreeSet::new();
    context
        .steps
        .iter()
        .map(|step| {
            if step.fixed_destination.is_some() {
                false
            } else if let Some(anchor_id) = step.anchor_id {
                seen_anchors.insert(anchor_id)
            } else {
                true
            }
        })
        .collect()
}

fn forward_feasibility(
    graph: &OdGraph,
    context: &Context,
    domains: &[Vec<usize>],
    parameters: Parameters,
) -> Result<ForwardFeasibility, SamplerError> {
    let initial = *graph.zone_index.get(&context.initial_zone).ok_or_else(|| {
        SamplerError::InvalidInput(format!(
            "context {} uses unknown initial zone {}",
            context.context_id, context.initial_zone
        ))
    })?;
    let layers = context.steps.len();
    let mut stats = FeasibilityStats::default();
    let mut reachable_by_layer = Vec::with_capacity(layers);

    // The first pair is reachable when the initial home location has an OD
    // edge to the first destination. Its activity duration is checked when
    // the following destination becomes known.
    let mut first = vec![0_u8; domains[0].len()];
    for (current_index, &current) in domains[0].iter().enumerate() {
        stats.forward_pair_states += 1;
        if graph.edge_to(initial, current).is_some() {
            first[current_index] = 1;
            stats.forward_reachable_pair_states += 1;
        }
    }
    reachable_by_layer.push(first);

    for layer in 1..layers {
        let prior_step = context.steps[layer - 1];
        let previous_domain = &domains[layer - 1];
        let current_domain = &domains[layer];
        let preprevious_domain: &[usize] = if layer == 1 {
            std::slice::from_ref(&initial)
        } else {
            &domains[layer - 2]
        };
        let prior_reachable = &reachable_by_layer[layer - 1];
        let mut current_reachable = vec![0_u8; previous_domain.len() * current_domain.len()];
        let mut current_index_by_destination = vec![usize::MAX; graph.zone_ids.len()];
        for (current_index, &current) in current_domain.iter().enumerate() {
            current_index_by_destination[current] = current_index;
        }

        for (previous_index, &previous) in previous_domain.iter().enumerate() {
            // For an existential feasibility pass, every valid prefix reaching
            // `previous` is interchangeable except for its adjusted arrival.
            // Keeping the earliest arrival is exact: it leaves the greatest
            // possible duration for the activity that ends at `previous`.
            let earliest_arrival = if parameters.update_plan_timings {
                let mut earliest = f64::INFINITY;
                for (preprevious_index, &preprevious) in preprevious_domain.iter().enumerate() {
                    if prior_reachable[preprevious_index * previous_domain.len() + previous_index]
                        == 0
                    {
                        continue;
                    }
                    let Some(incoming) = graph.edge_to(preprevious, previous) else {
                        continue;
                    };
                    let Some((_, arrival)) = adjusted_times(prior_step, incoming) else {
                        continue;
                    };
                    earliest = earliest.min(arrival);
                }
                earliest.is_finite().then_some(earliest)
            } else {
                (0..preprevious_domain.len())
                    .any(|preprevious_index| {
                        prior_reachable[preprevious_index * previous_domain.len() + previous_index]
                            != 0
                    })
                    .then_some(0.0)
            };
            let Some(earliest_arrival) = earliest_arrival else {
                continue;
            };

            if current_domain.len() <= 8 {
                // A singleton fixed destination or conditioned anchor is
                // cheaper to probe directly than to scan a time-ordered row.
                for (current_index, &current) in current_domain.iter().enumerate() {
                    stats.forward_pair_states += 1;
                    let Some(outgoing) = graph.edge_to(previous, current) else {
                        continue;
                    };
                    let following_departure = if parameters.update_plan_timings {
                        let Some((departure, _)) = adjusted_times(context.steps[layer], outgoing)
                        else {
                            continue;
                        };
                        departure
                    } else {
                        0.0
                    };
                    let duration = if parameters.update_plan_timings {
                        following_departure - earliest_arrival
                    } else {
                        prior_step.duration_per_person
                    };
                    stats.duration_checks += 1;
                    if !duration_is_feasible(prior_step, duration) {
                        stats.duration_infeasible += 1;
                        continue;
                    }
                    current_reachable[previous_index * current_domain.len() + current_index] = 1;
                    stats.forward_reachable_pair_states += 1;
                }
                continue;
            }

            // The duration of the previous activity is non-increasing with
            // this outgoing travel time. Once it becomes infeasible, every
            // later edge in this time-ordered row is infeasible as well.
            stats.forward_pair_states += current_domain.len() as u64;
            for outgoing in graph.outgoing_by_time(previous) {
                stats.forward_time_edge_scans += 1;
                let following_departure = if parameters.update_plan_timings {
                    let Some((departure, _)) = adjusted_times(context.steps[layer], outgoing)
                    else {
                        continue;
                    };
                    departure
                } else {
                    0.0
                };
                let duration = if parameters.update_plan_timings {
                    following_departure - earliest_arrival
                } else {
                    prior_step.duration_per_person
                };
                stats.duration_checks += 1;
                if !duration_is_feasible(prior_step, duration) {
                    stats.duration_infeasible += 1;
                    stats.forward_time_cutoffs += 1;
                    break;
                }
                let current_index = current_index_by_destination[outgoing.destination];
                if current_index == usize::MAX {
                    continue;
                }
                current_reachable[previous_index * current_domain.len() + current_index] = 1;
                stats.forward_reachable_pair_states += 1;
            }
        }
        reachable_by_layer.push(current_reachable);
    }

    Ok(ForwardFeasibility {
        reachable_by_layer,
        stats,
    })
}

fn solve_conditional(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    context: &Context,
    domains: &[Vec<usize>],
    first_choices: &[bool],
    parameters: Parameters,
) -> Result<ConditionalResult, SamplerError> {
    let initial = *graph.zone_index.get(&context.initial_zone).ok_or_else(|| {
        SamplerError::InvalidInput(format!(
            "context {} uses unknown initial zone {}",
            context.context_id, context.initial_zone
        ))
    })?;
    let layers = context.steps.len();
    if layers == 0 {
        return Err(SamplerError::InvalidInput(format!(
            "context {} has no steps",
            context.context_id
        )));
    }

    // The boolean pass is cheaper than scoring every ternary transition. Its
    // states are exact: a state is kept only when a feasible prefix from home
    // exists. Intersecting it with the backward continuation therefore cannot
    // remove a feasible complete plan.
    let forward = parameters
        .use_bidirectional_feasibility
        .then(|| forward_feasibility(graph, context, domains, parameters))
        .transpose()?;

    // Each message is indexed by (previous destination, current destination).
    // The following destination is streamed through the log-sum-exp, keeping
    // memory quadratic even though the rigidity-aware factor is ternary.
    let last = layers - 1;
    let previous_domain: &[usize] = if last == 0 {
        std::slice::from_ref(&initial)
    } else {
        &domains[last - 1]
    };
    let mut next_values = vec![f64::NEG_INFINITY; previous_domain.len() * domains[last].len()];
    let mut stats = forward
        .as_ref()
        .map(|forward| forward.stats)
        .unwrap_or_default();
    let last_step = context.steps[last];
    let terminal_home = last_step.activity_id == 0 && last_step.fixed_destination.is_some();
    for (previous_index, &previous) in previous_domain.iter().enumerate() {
        for (current_index, &current) in domains[last].iter().enumerate() {
            stats.pair_states += 1;
            if forward.as_ref().is_some_and(|forward| {
                forward.reachable_by_layer[last]
                    [previous_index * domains[last].len() + current_index]
                    == 0
            }) {
                continue;
            }
            let Some(edge) = graph.edge_to(previous, current) else {
                continue;
            };
            let Some(attraction) =
                attraction(destinations, last_step, current, first_choices[last])
            else {
                continue;
            };
            let duration = if terminal_home {
                // The final home row marks the return-home trip. Mobility
                // values the wrapped morning and overnight home time once at
                // plan level, so this boundary has no activity-duration term.
                last_step.min_activity_time
            } else {
                let duration = if parameters.update_plan_timings {
                    let Some((_, arrival)) = adjusted_times(last_step, edge) else {
                        continue;
                    };
                    last_step.next_departure_time - arrival
                } else {
                    last_step.duration_per_person
                };
                stats.duration_checks += 1;
                if !duration_is_feasible(last_step, duration) {
                    stats.duration_infeasible += 1;
                    continue;
                }
                duration
            };
            stats.scored_transitions += 1;
            let wrapped_home_adjustment = if terminal_home && parameters.update_plan_timings {
                let Some((_, adjusted_arrival)) = adjusted_times(last_step, edge) else {
                    continue;
                };
                -parameters.logit_scale
                    * parameters.wrapped_home_time_shadow_price
                    * (adjusted_arrival
                        - last_step
                            .arrival_time
                            .expect("reference contexts include arrival times"))
            } else {
                0.0
            };
            next_values[previous_index * domains[last].len() + current_index] = activity_log_weight(
                last_step,
                edge,
                destination_value(destinations, last_step, current),
                duration,
                attraction,
                parameters,
            )
                + wrapped_home_adjustment;
            stats.feasible_pair_states += 1;
            stats.corridor_pair_states += 1;
        }
    }

    for layer in (0..last).rev() {
        let previous_domain: &[usize] = if layer == 0 {
            std::slice::from_ref(&initial)
        } else {
            &domains[layer - 1]
        };
        let current_domain = &domains[layer];
        let following_domain = &domains[layer + 1];
        let mut current_values =
            vec![f64::NEG_INFINITY; previous_domain.len() * current_domain.len()];
        let active_following = (0..current_domain.len())
            .map(|current_index| {
                (0..following_domain.len())
                    .filter(|&following_index| {
                        next_values[current_index * following_domain.len() + following_index]
                            .is_finite()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut following_departures = vec![None; current_domain.len() * following_domain.len()];
        if parameters.update_plan_timings {
            for (current_index, &current) in current_domain.iter().enumerate() {
                for &following_index in &active_following[current_index] {
                    let following = following_domain[following_index];
                    following_departures
                        [current_index * following_domain.len() + following_index] = graph
                        .edge_to(current, following)
                        .and_then(|edge| adjusted_times(context.steps[layer + 1], edge))
                        .map(|(departure, _)| departure);
                }
            }
        }
        let step = context.steps[layer];
        let mut previous_index_by_destination = vec![usize::MAX; graph.zone_ids.len()];
        for (previous_index, &previous) in previous_domain.iter().enumerate() {
            previous_index_by_destination[previous] = previous_index;
        }
        let mut current_duration_possible = vec![true; current_values.len()];
        if parameters.update_plan_timings && previous_domain.len() > 8 {
            current_duration_possible.fill(false);
            for (current_index, &current) in current_domain.iter().enumerate() {
                let latest_following_departure = active_following[current_index]
                    .iter()
                    .filter_map(|&following_index| {
                        following_departures
                            [current_index * following_domain.len() + following_index]
                    })
                    .max_by(f64::total_cmp);
                let Some(latest_following_departure) = latest_following_departure else {
                    continue;
                };

                // The current activity loses duration as its incoming trip
                // gets longer. An incoming time-ordered scan can therefore
                // stop once even the latest feasible following departure no
                // longer leaves enough time for this activity.
                for (previous, incoming) in graph.incoming_by_time(current) {
                    stats.backward_time_edge_scans += 1;
                    let Some((_, adjusted_arrival)) = adjusted_times(step, incoming) else {
                        continue;
                    };
                    let duration = latest_following_departure - adjusted_arrival;
                    stats.duration_checks += 1;
                    if !duration_is_feasible(step, duration) {
                        stats.duration_infeasible += 1;
                        stats.backward_time_cutoffs += 1;
                        break;
                    }
                    let previous_index = previous_index_by_destination[previous];
                    if previous_index != usize::MAX {
                        current_duration_possible
                            [previous_index * current_domain.len() + current_index] = true;
                    }
                }
            }
        }
        let mut alternatives = vec![f64::NEG_INFINITY; following_domain.len()];
        for (previous_index, &previous) in previous_domain.iter().enumerate() {
            for (current_index, &current) in current_domain.iter().enumerate() {
                stats.pair_states += 1;
                if forward.as_ref().is_some_and(|forward| {
                    forward.reachable_by_layer[layer]
                        [previous_index * current_domain.len() + current_index]
                        == 0
                }) {
                    continue;
                }
                if !current_duration_possible[previous_index * current_domain.len() + current_index]
                {
                    continue;
                }
                if active_following[current_index].is_empty() {
                    continue;
                }
                let Some(incoming) = graph.edge_to(previous, current) else {
                    continue;
                };
                let Some(attraction) =
                    attraction(destinations, step, current, first_choices[layer])
                else {
                    continue;
                };
                let value = destination_value(destinations, step, current);
                let coefficient = if parameters.use_shadow_prices {
                    value.country_value_coefficient * step.value_of_time + value.shadow_price
                } else {
                    value.country_value_coefficient * value.saturation_utility * step.value_of_time
                };
                let (adjusted_departure, adjusted_arrival) = if parameters.update_plan_timings {
                    let Some((departure, arrival)) = adjusted_times(step, incoming) else {
                        continue;
                    };
                    (departure, arrival)
                } else {
                    (step.departure_time, 0.0)
                };
                let wrapped_home_adjustment = if layer == 0 && parameters.update_plan_timings {
                    parameters.logit_scale
                        * parameters.wrapped_home_time_shadow_price
                        * (adjusted_departure - step.departure_time)
                } else {
                    0.0
                };
                let base =
                    attraction - parameters.logit_scale * incoming.cost + wrapped_home_adjustment;
                let utility_scale =
                    parameters.logit_scale * coefficient * step.mean_duration_per_person;
                let mut maximum = f64::NEG_INFINITY;
                for &following_index in &active_following[current_index] {
                    let continuation =
                        next_values[current_index * following_domain.len() + following_index];
                    let duration = if parameters.update_plan_timings {
                        let Some(departure) = following_departures
                            [current_index * following_domain.len() + following_index]
                        else {
                            alternatives[following_index] = f64::NEG_INFINITY;
                            continue;
                        };
                        departure - adjusted_arrival
                    } else {
                        step.duration_per_person
                    };
                    stats.duration_checks += 1;
                    if !duration_is_feasible(step, duration) {
                        stats.duration_infeasible += 1;
                        alternatives[following_index] = f64::NEG_INFINITY;
                        continue;
                    }
                    stats.scored_transitions += 1;
                    let duration_factor = (duration / step.min_activity_time).ln().max(0.0);
                    let score = base + utility_scale * duration_factor + continuation;
                    alternatives[following_index] = score;
                    maximum = maximum.max(score);
                }
                if maximum.is_finite() {
                    let sum = active_following[current_index]
                        .iter()
                        .map(|&index| (alternatives[index] - maximum).exp())
                        .sum::<f64>();
                    current_values[previous_index * current_domain.len() + current_index] =
                        maximum + sum.ln();
                    stats.feasible_pair_states += 1;
                    stats.corridor_pair_states += 1;
                }
            }
        }
        next_values = current_values;
    }

    let log_partition = next_values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, logaddexp);
    let mut first_probability = vec![0.0; graph.zone_ids.len()];
    if log_partition.is_finite() {
        for (&destination, &value) in domains[0].iter().zip(&next_values) {
            first_probability[destination] = (value - log_partition).exp();
        }
    }
    Ok(ConditionalResult {
        log_partition,
        first_probability,
        stats,
    })
}

fn solve_context(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    context: &Context,
    parameters: Parameters,
) -> Result<ContextResult, SamplerError> {
    let mut anchor_visits = BTreeMap::new();
    for anchor_id in context.steps.iter().filter_map(|step| {
        if step.fixed_destination.is_none() {
            step.anchor_id
        } else {
            None
        }
    }) {
        *anchor_visits.entry(anchor_id).or_insert(0_usize) += 1;
    }
    // A single anchor visit is an ordinary destination choice. Conditioning
    // is only needed when several visits must share the same destination.
    let anchor_ids = anchor_visits
        .into_iter()
        .filter_map(|(anchor_id, visits)| (visits > 1).then_some(anchor_id))
        .collect::<BTreeSet<_>>();
    if anchor_ids.len() > 1 {
        return Err(SamplerError::InvalidInput(format!(
            "experimental second-order solver supports at most one variable anchor in context {}",
            context.context_id
        )));
    }
    let first_choices = first_choice_by_layer(context);
    let Some(&anchor_id) = anchor_ids.first() else {
        let domains = build_domains(graph, destinations, context, None, None)?;
        return Ok(ContextResult {
            context_id: context.context_id,
            conditional: solve_conditional(
                graph,
                destinations,
                context,
                &domains,
                &first_choices,
                parameters,
            )?,
            anchor_conditions: 0,
        });
    };

    let anchor_activity = context
        .steps
        .iter()
        .find(|step| step.anchor_id == Some(anchor_id))
        .expect("the indexed anchor occurs")
        .activity_id;
    if context
        .steps
        .iter()
        .any(|step| step.anchor_id == Some(anchor_id) && step.activity_id != anchor_activity)
    {
        return Err(SamplerError::InvalidInput(format!(
            "context {} uses anchor id {} for several activities",
            context.context_id, anchor_id
        )));
    }
    let anchor_domain = destinations
        .domain(anchor_activity)
        .filter(|domain| !domain.is_empty())
        .ok_or(SamplerError::NoFeasibleSequence {
            context_id: context.context_id,
            origin: context.initial_zone,
        })?;
    let mut conditional_results = Vec::with_capacity(anchor_domain.len());
    let mut total_log_partition = f64::NEG_INFINITY;
    for &anchor_destination in anchor_domain {
        let domains = build_domains(
            graph,
            destinations,
            context,
            Some(anchor_id),
            Some(anchor_destination),
        )?;
        let result = solve_conditional(
            graph,
            destinations,
            context,
            &domains,
            &first_choices,
            parameters,
        )?;
        total_log_partition = logaddexp(total_log_partition, result.log_partition);
        conditional_results.push(result);
    }
    let mut first_probability = vec![0.0; graph.zone_ids.len()];
    let mut stats = FeasibilityStats::default();
    if total_log_partition.is_finite() {
        for result in &conditional_results {
            let weight = (result.log_partition - total_log_partition).exp();
            for (total, &conditional) in first_probability.iter_mut().zip(&result.first_probability)
            {
                *total += weight * conditional;
            }
        }
    }
    for result in conditional_results {
        stats.add(result.stats);
    }
    Ok(ContextResult {
        context_id: context.context_id,
        conditional: ConditionalResult {
            log_partition: total_log_partition,
            first_probability,
            stats,
        },
        anchor_conditions: anchor_domain.len(),
    })
}

pub fn solve_second_order_all(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    contexts: &[Context],
    parameters: Parameters,
    n_threads: Option<usize>,
) -> Result<SecondOrderResult, SamplerError> {
    let started = Instant::now();
    let solve = || {
        contexts
            .par_iter()
            .map(|context| solve_context(graph, destinations, context, parameters))
            .collect::<Vec<_>>()
    };
    let results = if let Some(n_threads) = n_threads {
        if n_threads == 0 {
            return Err(SamplerError::InvalidInput(
                "n_threads must be positive".to_string(),
            ));
        }
        ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .map_err(|error| SamplerError::InvalidInput(error.to_string()))?
            .install(solve)
    } else {
        solve()
    };

    let mut context_ids = Vec::with_capacity(contexts.len());
    let mut log_partitions = Vec::with_capacity(contexts.len());
    let mut first_destination_probabilities =
        Vec::with_capacity(contexts.len() * graph.zone_ids.len());
    let mut anchor_conditions = 0;
    let mut infeasible_contexts = 0;
    let mut stats = FeasibilityStats::default();
    for result in results {
        match result {
            Ok(result) => {
                if !result.conditional.log_partition.is_finite() {
                    infeasible_contexts += 1;
                }
                context_ids.push(result.context_id);
                log_partitions.push(result.conditional.log_partition);
                first_destination_probabilities.extend(result.conditional.first_probability);
                stats.add(result.conditional.stats);
                anchor_conditions += result.anchor_conditions;
            }
            Err(SamplerError::NoFeasibleSequence { .. }) if parameters.skip_infeasible => {
                infeasible_contexts += 1;
            }
            Err(error) => return Err(error),
        }
    }
    Ok(SecondOrderResult {
        context_ids,
        log_partitions,
        first_destination_probabilities,
        zone_ids: graph.zone_ids.clone(),
        wall_time: started.elapsed(),
        anchor_conditions,
        infeasible_contexts,
        duration_checks: stats.duration_checks,
        duration_infeasible: stats.duration_infeasible,
        scored_transitions: stats.scored_transitions,
        pair_states: stats.pair_states,
        feasible_pair_states: stats.feasible_pair_states,
        forward_pair_states: stats.forward_pair_states,
        forward_reachable_pair_states: stats.forward_reachable_pair_states,
        forward_time_edge_scans: stats.forward_time_edge_scans,
        forward_time_cutoffs: stats.forward_time_cutoffs,
        backward_time_edge_scans: stats.backward_time_edge_scans,
        backward_time_cutoffs: stats.backward_time_cutoffs,
        corridor_pair_states: stats.corridor_pair_states,
    })
}
