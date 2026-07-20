use rayon::prelude::*;
use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use crate::errors::SamplerError;
use crate::input::{Context, Step};
use crate::model::{DestinationIndex, DestinationValue, Edge, OdGraph};
use crate::output::{OutputRow, OutputTable};
use crate::profile::{ContextProfile, ProfileReport};

#[derive(Clone, Copy, Debug)]
pub struct Parameters {
    pub logit_scale: f64,
    pub update_plan_timings: bool,
    pub use_shadow_prices: bool,
    pub seed: u64,
    pub n_draws: u32,
    pub skip_infeasible: bool,
    pub wrapped_home_time_shadow_price: f64,
    pub use_bidirectional_feasibility: bool,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct StructuralStep {
    activity_id: u32,
    anchor_id: Option<u32>,
    is_fixed: bool,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct StructuralPlanKey {
    steps: Vec<StructuralStep>,
}

#[derive(Debug)]
struct CompiledAnchorPlan {
    anchor_slot_by_layer: Vec<Option<usize>>,
    first_anchor_visit_by_layer: Vec<bool>,
    domains: Vec<Vec<usize>>,
}

/// Structural plans are valid for the lifetime of one sampler. Mobility creates
/// a new sampler when iteration inputs change, so no cross-iteration
/// invalidation is needed.
#[derive(Debug, Default)]
pub struct IterationCache {
    anchor_plans: RwLock<HashMap<StructuralPlanKey, Arc<CompiledAnchorPlan>>>,
}

impl IterationCache {
    fn anchor_plan(
        &self,
        destinations: &DestinationIndex,
        context: &Context,
    ) -> Result<Arc<CompiledAnchorPlan>, SamplerError> {
        let key = StructuralPlanKey {
            steps: context
                .steps
                .iter()
                .map(|step| StructuralStep {
                    activity_id: step.activity_id,
                    anchor_id: step.anchor_id,
                    is_fixed: step.fixed_destination.is_some(),
                })
                .collect(),
        };
        if let Some(plan) = self
            .anchor_plans
            .read()
            .expect("the iteration cache lock is not poisoned")
            .get(&key)
        {
            return Ok(Arc::clone(plan));
        }

        let plan = Arc::new(compile_anchor_plan(destinations, context)?);
        let mut plans = self
            .anchor_plans
            .write()
            .expect("the iteration cache lock is not poisoned");
        Ok(Arc::clone(
            plans.entry(key).or_insert_with(|| Arc::clone(&plan)),
        ))
    }
}

#[inline]
pub(crate) fn local_log_weight(
    step: Step,
    edge: Edge,
    destination: DestinationValue,
    is_fixed: bool,
    parameters: Parameters,
) -> Option<f64> {
    let available_duration = if parameters.update_plan_timings {
        step.next_departure_time - step.departure_time - edge.time
    } else {
        step.duration_per_person
    };
    if available_duration < 0.0 {
        return None;
    }

    let activity_coefficient = if parameters.use_shadow_prices {
        destination.country_value_coefficient * step.value_of_time + destination.shadow_price
    } else {
        destination.country_value_coefficient * destination.saturation_utility * step.value_of_time
    };
    let duration_factor = (available_duration / step.min_activity_time).ln().max(0.0);
    let activity_utility = activity_coefficient * step.mean_duration_per_person * duration_factor;
    let attraction = if is_fixed {
        0.0
    } else if destination.log_opportunity_capacity.is_finite() {
        destination.log_opportunity_capacity
    } else {
        return None;
    };
    Some(attraction + parameters.logit_scale * (activity_utility - edge.cost))
}

#[inline]
pub(crate) fn logaddexp(left: f64, right: f64) -> f64 {
    if left == f64::NEG_INFINITY {
        return right;
    }
    if right == f64::NEG_INFINITY {
        return left;
    }
    let maximum = left.max(right);
    maximum + (-(left - right).abs()).exp().ln_1p()
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

fn score_anchor_assignment(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    context: &Context,
    parameters: Parameters,
    plan: &CompiledAnchorPlan,
    anchor_destinations: &[usize],
    mut local_weights: Option<&mut Vec<f64>>,
) -> Option<f64> {
    let mut origin = *graph.zone_index.get(&context.initial_zone)?;
    let mut total = 0.0;

    for (layer, step) in context.steps.iter().copied().enumerate() {
        let destination = if let Some(fixed_zone_id) = step.fixed_destination {
            *graph.zone_index.get(&fixed_zone_id)?
        } else {
            anchor_destinations[plan.anchor_slot_by_layer[layer]?]
        };
        let edge = graph.edge_to(origin, destination)?;
        let destination_value =
            fixed_destination_value(destinations.activity(step.activity_id), destination);
        let local = local_log_weight(
            step,
            edge,
            destination_value,
            !plan.first_anchor_visit_by_layer[layer],
            parameters,
        )?;
        total += local;
        if let Some(weights) = local_weights.as_deref_mut() {
            weights.push(local);
        }
        origin = destination;
    }
    Some(total)
}

fn compile_anchor_plan(
    destinations: &DestinationIndex,
    context: &Context,
) -> Result<CompiledAnchorPlan, SamplerError> {
    let anchor_ids = context
        .steps
        .iter()
        .filter_map(|step| {
            if step.fixed_destination.is_none() {
                step.anchor_id
            } else {
                None
            }
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    if anchor_ids.len() > 2 {
        return Err(SamplerError::InvalidInput(format!(
            "context {} contains {} variable anchor types; at most two are supported",
            context.context_id,
            anchor_ids.len()
        )));
    }

    let mut anchor_slot_by_layer = Vec::with_capacity(context.steps.len());
    let mut first_anchor_visit_by_layer = vec![false; context.steps.len()];
    let mut seen_anchors = BTreeSet::new();
    for (layer, step) in context.steps.iter().enumerate() {
        let slot = if step.fixed_destination.is_some() {
            None
        } else {
            let anchor_id = step.anchor_id.ok_or_else(|| {
                SamplerError::InvalidInput(format!(
                    "context {} contains a variable step without an anchor id",
                    context.context_id
                ))
            })?;
            first_anchor_visit_by_layer[layer] = seen_anchors.insert(anchor_id);
            Some(
                anchor_ids
                    .binary_search(&anchor_id)
                    .expect("every variable anchor id is indexed"),
            )
        };
        anchor_slot_by_layer.push(slot);
    }

    let mut domains = Vec::with_capacity(anchor_ids.len());
    for &anchor_id in &anchor_ids {
        let activity_id = context
            .steps
            .iter()
            .find(|step| step.anchor_id == Some(anchor_id))
            .expect("every indexed anchor occurs in the plan")
            .activity_id;
        if context
            .steps
            .iter()
            .any(|step| step.anchor_id == Some(anchor_id) && step.activity_id != activity_id)
        {
            return Err(SamplerError::InvalidInput(format!(
                "context {} uses anchor id {} for several activities",
                context.context_id, anchor_id
            )));
        }
        let domain = destinations
            .domain(activity_id)
            .ok_or(SamplerError::NoFeasibleSequence {
                context_id: context.context_id,
                origin: context.initial_zone,
            })?;
        if domain.is_empty() {
            return Err(SamplerError::NoFeasibleSequence {
                context_id: context.context_id,
                origin: context.initial_zone,
            });
        }
        domains.push(domain.to_vec());
    }
    Ok(CompiledAnchorPlan {
        anchor_slot_by_layer,
        first_anchor_visit_by_layer,
        domains,
    })
}

fn for_each_anchor_assignment(domains: &[Vec<usize>], mut visit: impl FnMut(&[usize])) {
    match domains {
        [] => visit(&[]),
        [first] => {
            for &destination in first {
                visit(&[destination]);
            }
        }
        [first, second] => {
            for &first_destination in first {
                for &second_destination in second {
                    visit(&[first_destination, second_destination]);
                }
            }
        }
        _ => unreachable!("anchor contexts are limited to two variable anchor types"),
    }
}

#[inline]
pub(crate) fn alternative_gumbel(
    seed: u64,
    context_id: u64,
    draw_id: u32,
    alternative_index: u64,
) -> f64 {
    let mut value = seed
        ^ context_id.wrapping_mul(0x9E3779B97F4A7C15)
        ^ u64::from(draw_id).wrapping_mul(0xBF58476D1CE4E5B9)
        ^ alternative_index.wrapping_mul(0x94D049BB133111EB);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
    value ^= value >> 31;
    let unit = ((value >> 11) as f64 + 0.5) / ((1u64 << 53) as f64);
    -(-unit.ln()).ln()
}

fn sample_unique_anchor_context(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    context: &Context,
    plan: &CompiledAnchorPlan,
    parameters: Parameters,
) -> Result<OutputTable, SamplerError> {
    // Gumbel-max draws exactly from the categorical distribution while each
    // complete anchor assignment is scored only once.
    let mut best_perturbed_scores = vec![f64::NEG_INFINITY; parameters.n_draws as usize];
    let mut selected_assignments = vec![Vec::new(); parameters.n_draws as usize];
    let mut alternative_index = 0_u64;
    for_each_anchor_assignment(&plan.domains, |assignment| {
        let Some(assignment_score) = score_anchor_assignment(
            graph,
            destinations,
            context,
            parameters,
            plan,
            assignment,
            None,
        ) else {
            alternative_index += 1;
            return;
        };
        for draw_id in 1..=parameters.n_draws {
            let perturbed_score = assignment_score
                + alternative_gumbel(
                    parameters.seed,
                    context.context_id,
                    draw_id,
                    alternative_index,
                );
            let draw_index = (draw_id - 1) as usize;
            if perturbed_score > best_perturbed_scores[draw_index] {
                best_perturbed_scores[draw_index] = perturbed_score;
                selected_assignments[draw_index] = assignment.to_vec();
            }
        }
        alternative_index += 1;
    });
    if best_perturbed_scores.iter().any(|score| !score.is_finite()) {
        return Err(SamplerError::NoFeasibleSequence {
            context_id: context.context_id,
            origin: context.initial_zone,
        });
    }

    let initial_zone = graph.zone_index[&context.initial_zone];
    let mut output = OutputTable::default();
    for (draw_index, assignment) in selected_assignments.iter().enumerate() {
        let mut local_weights = Vec::with_capacity(context.steps.len());
        score_anchor_assignment(
            graph,
            destinations,
            context,
            parameters,
            plan,
            assignment,
            Some(&mut local_weights),
        )
        .expect("selected anchor assignments are feasible");
        let mut suffix_values = vec![0.0; local_weights.len()];
        let mut suffix = 0.0;
        for layer in (0..local_weights.len()).rev() {
            suffix += local_weights[layer];
            suffix_values[layer] = suffix;
        }

        let mut origin = initial_zone;
        for (layer, step) in context.steps.iter().copied().enumerate() {
            let destination = if let Some(fixed_zone_id) = step.fixed_destination {
                graph.zone_index[&fixed_zone_id]
            } else {
                assignment[plan.anchor_slot_by_layer[layer]
                    .expect("variable anchors have a compiled slot")]
            };
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
    Ok(output)
}

fn context_uses_unique_anchors(context: &Context) -> bool {
    context
        .steps
        .iter()
        .all(|step| step.fixed_destination.is_some() || step.anchor_id.is_some())
}

pub fn sample_all(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    cache: &IterationCache,
    contexts: &[Context],
    parameters: Parameters,
    n_threads: Option<usize>,
) -> Result<OutputTable, SamplerError> {
    let plans = contexts
        .iter()
        .map(|context| {
            if context_uses_unique_anchors(context) {
                cache.anchor_plan(destinations, context).map(Some)
            } else {
                Ok(None)
            }
        })
        .collect::<Result<Vec<_>, SamplerError>>()?;
    let compute = || {
        contexts
            .par_iter()
            .zip(plans.par_iter())
            .map(|(context, plan)| {
                if let Some(plan) = plan {
                    sample_unique_anchor_context(graph, destinations, context, plan, parameters)
                } else {
                    crate::factor_tree::sample_tree_context(
                        graph,
                        destinations,
                        context,
                        parameters,
                    )
                }
            })
            .collect::<Vec<_>>()
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
            .install(compute)
    } else {
        compute()
    };

    let mut output = OutputTable::default();
    for table in tables {
        match table {
            Ok(table) => output.extend(table),
            Err(SamplerError::NoFeasibleSequence { .. }) if parameters.skip_infeasible => {}
            Err(SamplerError::CyclicContext { .. }) if parameters.skip_infeasible => {}
            Err(error) => return Err(error),
        }
    }
    Ok(output)
}

pub fn sample_all_with_profile(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    cache: &IterationCache,
    contexts: &[Context],
    parameters: Parameters,
    n_threads: Option<usize>,
) -> Result<(OutputTable, ProfileReport), SamplerError> {
    let plan_started = Instant::now();
    let plans = contexts
        .iter()
        .map(|context| {
            if context_uses_unique_anchors(context) {
                cache.anchor_plan(destinations, context).map(Some)
            } else {
                Ok(None)
            }
        })
        .collect::<Result<Vec<_>, SamplerError>>()?;
    let plan_build = plan_started.elapsed();

    let compute = || {
        contexts
            .par_iter()
            .zip(plans.par_iter())
            .map(|(context, plan)| {
                let started = Instant::now();
                let mut profile = ContextProfile::default();
                let result = if let Some(plan) = plan {
                    sample_unique_anchor_context(graph, destinations, context, plan, parameters)
                } else {
                    crate::factor_tree::sample_tree_context_with_profile(
                        graph,
                        destinations,
                        context,
                        parameters,
                        &mut profile,
                    )
                };
                profile.total = started.elapsed();
                (result, profile, plan.is_some())
            })
            .collect::<Vec<_>>()
    };
    let sampling_started = Instant::now();
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
            .install(compute)
    } else {
        compute()
    };
    let sampling_wall = sampling_started.elapsed();

    let merge_started = Instant::now();
    let mut output = OutputTable::default();
    let mut report = ProfileReport {
        plan_build,
        sampling_wall,
        contexts: contexts.len() as u64,
        input_steps: contexts
            .iter()
            .map(|context| context.steps.len() as u64)
            .sum(),
        ..ProfileReport::default()
    };
    for (table, context_profile, is_anchor) in tables {
        report.add_context(&context_profile, is_anchor);
        if is_anchor {
            report.anchor_contexts += 1;
        } else {
            report.tree_contexts += 1;
        }
        match table {
            Ok(table) => {
                report.successful_contexts += 1;
                output.extend(table);
            }
            Err(SamplerError::NoFeasibleSequence { .. }) if parameters.skip_infeasible => {
                report.infeasible_contexts += 1;
            }
            Err(SamplerError::CyclicContext { .. }) if parameters.skip_infeasible => {
                report.cyclic_contexts += 1;
            }
            Err(error) => return Err(error),
        }
    }
    report.output_rows = output.context_id.len() as u64;
    report.output_merge = merge_started.elapsed();
    Ok((output, report))
}
