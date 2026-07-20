//! Bounded sequential proposal sampler for complete rigidity-aware plans.
//!
//! The exact ternary solver remains the oracle. This module never enumerates a
//! destination domain per particle: candidate sets are the bounded union of a
//! pre-ranked activity list, nearby OD edges, fixed/anchor look-ahead, and two
//! deterministic exploration draws.
//!
//! Status: baseline/fallback experiment. The public `sample_particles` API is
//! retained while the bidirectional top-K redesign is developed.

use std::collections::{BTreeMap, BTreeSet};

use rayon::prelude::*;

use crate::errors::SamplerError;
use crate::input::Context;
use crate::model::{DestinationIndex, OdGraph};
use crate::output::{OutputRow, OutputTable};
use crate::sampler::{alternative_gumbel, fixed_destination_value, Parameters};
use crate::ternary_reference::{
    activity_log_weight, adjusted_times, build_problem, score_zones, ReferenceProblem,
};

#[derive(Clone)]
struct Particle {
    zones: Vec<usize>,
    anchors: BTreeMap<u32, usize>,
    log_proposal: f64,
}

#[derive(Clone)]
struct CompletedParticle {
    zones: Vec<usize>,
    score: f64,
    log_proposal: f64,
}

#[derive(Debug, Default)]
pub struct ParticleReport {
    pub contexts: u64,
    pub candidate_evaluations: u64,
    pub locally_infeasible_candidates: u64,
    pub completed_particles: u64,
    pub selected_plans: u64,
    pub infeasible_contexts: u64,
    pub effective_sample_size_sum: f64,
    pub retry_attempts: u64,
    pub recovered_contexts: u64,
    pub context_reports: Vec<ParticleContextReport>,
}

#[derive(Debug, Clone)]
pub struct ParticleContextReport {
    pub context_id: u64,
    pub candidate_evaluations: u64,
    pub locally_infeasible_candidates: u64,
    pub completed_particles: u64,
    pub selected_plans: u64,
    pub effective_sample_size: f64,
    pub first_failure_layer: Option<u32>,
    pub failure_reason: Option<&'static str>,
    pub candidate_set_size_at_failure: Option<u64>,
    pub domain_locally_feasible_candidates: Option<u64>,
    pub retry_attempts: u32,
    pub recovered_by_retry: bool,
}

impl ParticleReport {
    fn add(&mut self, other: &Self) {
        self.contexts += other.contexts;
        self.candidate_evaluations += other.candidate_evaluations;
        self.locally_infeasible_candidates += other.locally_infeasible_candidates;
        self.completed_particles += other.completed_particles;
        self.selected_plans += other.selected_plans;
        self.infeasible_contexts += other.infeasible_contexts;
        self.effective_sample_size_sum += other.effective_sample_size_sum;
        self.retry_attempts += other.retry_attempts;
        self.recovered_contexts += other.recovered_contexts;
        self.context_reports
            .extend(other.context_reports.iter().cloned());
    }
}

fn logaddexp(left: f64, right: f64) -> f64 {
    if left == f64::NEG_INFINITY {
        return right;
    }
    if right == f64::NEG_INFINITY {
        return left;
    }
    let maximum = left.max(right);
    maximum + (-(left - right).abs()).exp().ln_1p()
}

fn deterministic_index(
    seed: u64,
    context_id: u64,
    particle: usize,
    layer: usize,
    draw: u64,
    len: usize,
) -> usize {
    let mut value = seed
        ^ context_id.wrapping_mul(0x9E3779B97F4A7C15)
        ^ (particle as u64).wrapping_mul(0xBF58476D1CE4E5B9)
        ^ (layer as u64).wrapping_mul(0x94D049BB133111EB)
        ^ draw.wrapping_mul(0xD6E8FEB86659FD93);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
    value ^= value >> 31;
    (value as usize) % len
}

fn retry_seed(seed: u64, retry_attempt: u32) -> u64 {
    seed ^ u64::from(retry_attempt).wrapping_mul(0xD6E8FEB86659FD93)
}

fn next_fixed_or_anchor(
    context: &Context,
    particle: &Particle,
    layer: usize,
    graph: &OdGraph,
) -> Option<usize> {
    let next = context.steps.get(layer + 1)?;
    next.fixed_destination
        .map(|zone| graph.zone_index[&zone])
        .or_else(|| {
            next.anchor_id
                .and_then(|anchor| particle.anchors.get(&anchor).copied())
        })
}

fn bounded_candidates(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    context: &Context,
    particle: &Particle,
    layer: usize,
    particle_index: usize,
    candidate_count: usize,
    seed: u64,
) -> Result<Vec<usize>, SamplerError> {
    let step = context.steps[layer];
    if let Some(fixed) = step.fixed_destination {
        return Ok(vec![graph.zone_index[&fixed]]);
    }
    if let Some(anchor) = step
        .anchor_id
        .and_then(|anchor| particle.anchors.get(&anchor))
    {
        return Ok(vec![*anchor]);
    }

    let domain = destinations
        .domain(step.activity_id)
        .ok_or(SamplerError::NoFeasibleSequence {
            context_id: context.context_id,
            origin: context.initial_zone,
        })?;
    let origin = particle
        .zones
        .last()
        .copied()
        .unwrap_or(graph.zone_index[&context.initial_zone]);
    let values = destinations
        .activity(step.activity_id)
        .expect("domain has an activity table");
    let mut candidates = BTreeSet::new();
    for &zone in destinations
        .attractive(step.activity_id)
        .unwrap_or(&[])
        .iter()
        .take(candidate_count)
    {
        candidates.insert(zone);
    }
    // Cost-ordered OD indexes make this a bounded generalized-cost candidate
    // source without a domain scan. Retain only valid destinations.
    for edge in graph.outgoing_by_cost(origin).take(candidate_count) {
        if values[edge.destination].is_some_and(|value| value.log_opportunity_capacity.is_finite())
        {
            candidates.insert(edge.destination);
        }
    }
    if let Some(next_zone) = next_fixed_or_anchor(context, particle, layer, graph) {
        for (zone, _) in graph.incoming_by_cost(next_zone).take(candidate_count) {
            if values[zone].is_some_and(|value| value.log_opportunity_capacity.is_finite()) {
                candidates.insert(zone);
            }
        }
    }
    // A tiny deterministic opportunity-domain exploration component prevents
    // the beam from becoming a fixed top-destination list.
    for draw in 0..2 {
        candidates.insert(
            domain[deterministic_index(
                seed,
                context.context_id,
                particle_index,
                layer,
                draw,
                domain.len(),
            )],
        );
    }
    Ok(candidates.into_iter().collect())
}

fn proposal_score(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    context: &Context,
    particle: &Particle,
    layer: usize,
    candidate: usize,
    parameters: Parameters,
) -> Option<f64> {
    let step = context.steps[layer];
    let origin = particle
        .zones
        .last()
        .copied()
        .unwrap_or(graph.zone_index[&context.initial_zone]);
    let edge = graph.edge_to(origin, candidate)?;
    let (adjusted_departure, adjusted_arrival) = adjusted_times(step, edge)?;
    // Choosing this leg fixes the previous activity's outgoing departure.
    // Reject it now when that makes the adjacent duration non-positive rather
    // than keeping a particle that can only fail at final plan scoring.
    if parameters.update_plan_timings && layer > 0 {
        let previous_step = context.steps[layer - 1];
        let previous_origin = if layer > 1 {
            particle.zones[layer - 2]
        } else {
            graph.zone_index[&context.initial_zone]
        };
        let previous_destination = particle.zones[layer - 1];
        let previous_edge = graph.edge_to(previous_origin, previous_destination)?;
        let (_, previous_adjusted_arrival) = adjusted_times(previous_step, previous_edge)?;
        if adjusted_departure - previous_adjusted_arrival <= 0.0 {
            return None;
        }
    }
    let terminal_fixed_home = layer + 1 == context.steps.len() && step.fixed_destination.is_some();
    // The terminal home period is a flexible day boundary. It still pays the
    // return-leg cost, but must not be rejected as an ordinary activity whose
    // duration is bounded by the following departure time.
    let provisional_duration = if terminal_fixed_home {
        step.duration_per_person.max(0.0)
    } else if parameters.update_plan_timings {
        // This is deliberately conservative local screening. The actual
        // adjacent duration is checked after the next destination is chosen.
        step.next_departure_time - adjusted_arrival
    } else {
        step.duration_per_person
    };
    if !terminal_fixed_home && provisional_duration <= 0.0 {
        return None;
    }
    let value = fixed_destination_value(destinations.activity(step.activity_id), candidate);
    let is_first_choice = step
        .anchor_id
        .is_none_or(|anchor| !particle.anchors.contains_key(&anchor));
    let attraction = if step.fixed_destination.is_some() || !is_first_choice {
        0.0
    } else {
        value.log_opportunity_capacity
    };
    if !attraction.is_finite() {
        return None;
    }
    let mut score = activity_log_weight(
        step,
        edge,
        value,
        provisional_duration,
        attraction,
        parameters,
    );
    if let Some(next_zone) = next_fixed_or_anchor(context, particle, layer, graph) {
        score -= parameters.logit_scale * graph.edge_to(candidate, next_zone)?.cost;
    } else if layer + 1 < context.steps.len() {
        // Variable next activities get a cheap optimistic one-leg look-ahead.
        score -= parameters.logit_scale
            * graph
                .outgoing_by_cost(candidate)
                .take(4)
                .map(|next_edge| next_edge.cost)
                .fold(f64::INFINITY, f64::min);
    }
    score.is_finite().then_some(score)
}

fn strictly_feasible(
    graph: &OdGraph,
    context: &Context,
    zones: &[usize],
    parameters: Parameters,
) -> bool {
    if !parameters.update_plan_timings {
        return true;
    }
    let mut adjusted_departures = Vec::with_capacity(zones.len());
    let mut adjusted_arrivals = Vec::with_capacity(zones.len());
    let mut origin = graph.zone_index[&context.initial_zone];
    for (&destination, &step) in zones.iter().zip(&context.steps) {
        let Some(edge) = graph.edge_to(origin, destination) else {
            return false;
        };
        let Some((departure, arrival)) = adjusted_times(step, edge) else {
            return false;
        };
        adjusted_departures.push(departure);
        adjusted_arrivals.push(arrival);
        origin = destination;
    }
    (0..zones.len()).all(|layer| {
        if layer + 1 == zones.len() && context.steps[layer].fixed_destination.is_some() {
            return true;
        }
        let next_departure = adjusted_departures
            .get(layer + 1)
            .copied()
            .unwrap_or(context.steps[layer].next_departure_time);
        next_departure - adjusted_arrivals[layer] > 0.0
    })
}

fn append_context_report(
    report: &mut ParticleReport,
    context: &Context,
    first_failure_layer: Option<usize>,
    failure_reason: Option<&'static str>,
    candidate_set_size_at_failure: Option<usize>,
    domain_locally_feasible_candidates: Option<usize>,
    retry_attempts: u32,
    recovered_by_retry: bool,
) {
    report.context_reports.push(ParticleContextReport {
        context_id: context.context_id,
        candidate_evaluations: report.candidate_evaluations,
        locally_infeasible_candidates: report.locally_infeasible_candidates,
        completed_particles: report.completed_particles,
        selected_plans: report.selected_plans,
        effective_sample_size: report.effective_sample_size_sum,
        first_failure_layer: first_failure_layer.map(|layer| layer as u32),
        failure_reason,
        candidate_set_size_at_failure: candidate_set_size_at_failure.map(|count| count as u64),
        domain_locally_feasible_candidates: domain_locally_feasible_candidates
            .map(|count| count as u64),
        retry_attempts,
        recovered_by_retry,
    });
}

fn append_plan(
    output: &mut OutputTable,
    graph: &OdGraph,
    destinations: &DestinationIndex,
    context: &Context,
    problem: &ReferenceProblem<'_>,
    zones: &[usize],
    parameters: Parameters,
    draw_id: u32,
    proposal_log_probability: f64,
    importance_log_weight: f64,
) {
    let (_, local_weights) = score_zones(graph, destinations, context, problem, zones, parameters)
        .expect("strictly feasible particle has a complete reference score");
    let mut suffix = 0.0;
    let mut suffixes = vec![0.0; local_weights.len()];
    for layer in (0..local_weights.len()).rev() {
        suffix += local_weights[layer];
        suffixes[layer] = suffix;
    }
    let mut origin = graph.zone_index[&context.initial_zone];
    for (layer, &destination) in zones.iter().enumerate() {
        output.push_particle(
            OutputRow {
                context_id: context.context_id,
                draw_id,
                layer: layer as u32,
                origin: graph.zone_ids[origin],
                destination: graph.zone_ids[destination],
                local_log_weight: local_weights[layer],
                total_log_weight: suffixes[layer],
            },
            proposal_log_probability,
            importance_log_weight,
        );
        origin = destination;
    }
}

fn sample_context(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    context: &Context,
    parameters: Parameters,
    particle_count: usize,
    candidate_count: usize,
    max_retries: u32,
) -> Result<(OutputTable, ParticleReport), SamplerError> {
    let mut report = ParticleReport {
        contexts: 1,
        ..ParticleReport::default()
    };
    let (problem, _) = match build_problem(graph, destinations, context) {
        Ok(problem) => problem,
        Err(SamplerError::NoFeasibleSequence { .. }) if parameters.skip_infeasible => {
            report.infeasible_contexts = 1;
            append_context_report(
                &mut report,
                context,
                Some(0),
                Some("no_feasible_activity_domain"),
                None,
                None,
                0,
                false,
            );
            return Ok((OutputTable::default(), report));
        }
        Err(error) => return Err(error),
    };
    let mut particles = vec![
        Particle {
            zones: Vec::new(),
            anchors: BTreeMap::new(),
            log_proposal: 0.0,
        };
        particle_count
    ];
    let mut first_failure_layer = None;
    let mut failure_reason = None;
    let mut candidate_set_size_at_failure = None;
    let mut domain_locally_feasible_candidates = None;
    let mut retry_attempts = 0u32;
    let mut layer = 0usize;
    // checkpoints[layer] is the particle frontier before choosing `layer`.
    let mut checkpoints = vec![particles.clone()];
    while layer < context.steps.len() {
        let sampling_seed = retry_seed(parameters.seed, retry_attempts);
        let mut next_particles = Vec::with_capacity(particles.len());
        let mut candidate_count_at_layer = 0usize;
        let mut scored_count_at_layer = 0usize;
        let mut domain_feasible_at_layer = 0usize;
        for (particle_index, particle) in particles.iter().enumerate() {
            let candidates = bounded_candidates(
                graph,
                destinations,
                context,
                particle,
                layer,
                particle_index,
                candidate_count,
                sampling_seed,
            )?;
            candidate_count_at_layer += candidates.len();
            let mut scored = Vec::with_capacity(candidates.len());
            for candidate in candidates {
                report.candidate_evaluations += 1;
                match proposal_score(
                    graph,
                    destinations,
                    context,
                    particle,
                    layer,
                    candidate,
                    parameters,
                ) {
                    Some(score) => scored.push((candidate, score)),
                    None => report.locally_infeasible_candidates += 1,
                }
            }
            scored_count_at_layer += scored.len();
            if scored.is_empty() {
                let step = context.steps[layer];
                if step.fixed_destination.is_none()
                    && step
                        .anchor_id
                        .is_none_or(|anchor| !particle.anchors.contains_key(&anchor))
                {
                    if let Some(domain) = destinations.domain(step.activity_id) {
                        domain_feasible_at_layer += domain
                            .iter()
                            .filter(|&&candidate| {
                                proposal_score(
                                    graph,
                                    destinations,
                                    context,
                                    particle,
                                    layer,
                                    candidate,
                                    parameters,
                                )
                                .is_some()
                            })
                            .count();
                    }
                }
            }
            if scored.is_empty() {
                continue;
            }
            let log_normalizer = scored.iter().fold(f64::NEG_INFINITY, |total, (_, score)| {
                logaddexp(total, *score)
            });
            let (candidate, score) = scored
                .iter()
                .enumerate()
                .max_by(|(left_index, (_, left)), (right_index, (_, right))| {
                    let left_value = left
                        + alternative_gumbel(
                            sampling_seed,
                            context.context_id,
                            particle_index as u32 + 1,
                            (layer * 10_000 + *left_index) as u64,
                        );
                    let right_value = right
                        + alternative_gumbel(
                            sampling_seed,
                            context.context_id,
                            particle_index as u32 + 1,
                            (layer * 10_000 + *right_index) as u64,
                        );
                    left_value.total_cmp(&right_value)
                })
                .map(|(_, value)| *value)
                .expect("non-empty scores");
            let mut selected = particle.clone();
            selected.zones.push(candidate);
            if let Some(anchor) = context.steps[layer].anchor_id {
                selected.anchors.entry(anchor).or_insert(candidate);
            }
            selected.log_proposal += score - log_normalizer;
            next_particles.push(selected);
        }
        particles = next_particles;
        if particles.is_empty() {
            // Retry from before the preceding choice. Repeating the failed
            // layer with the same parents cannot repair an empty candidate
            // set; changing the prior destination can.
            if layer > 0 && retry_attempts < max_retries {
                retry_attempts += 1;
                report.retry_attempts += 1;
                particles = checkpoints[layer - 1].clone();
                checkpoints.truncate(layer);
                layer -= 1;
                continue;
            }
            first_failure_layer = Some(layer);
            candidate_set_size_at_failure = Some(candidate_count_at_layer);
            domain_locally_feasible_candidates = Some(domain_feasible_at_layer);
            failure_reason = Some(if candidate_count_at_layer == 0 {
                "no_candidates"
            } else if scored_count_at_layer == 0 {
                "no_locally_feasible_candidate"
            } else {
                "all_particles_lost_after_sampling"
            });
            break;
        }
        checkpoints.push(particles.clone());
        layer += 1;
    }

    let mut final_duration_failures = 0usize;
    let mut completed = particles
        .into_iter()
        .filter(|particle| particle.zones.len() == context.steps.len())
        .filter(|particle| {
            let feasible = strictly_feasible(graph, context, &particle.zones, parameters);
            if !feasible {
                final_duration_failures += 1;
            }
            feasible
        })
        .filter_map(|particle| {
            score_zones(
                graph,
                destinations,
                context,
                &problem,
                &particle.zones,
                parameters,
            )
            .map(|(score, _)| CompletedParticle {
                zones: particle.zones,
                score,
                log_proposal: particle.log_proposal,
            })
        })
        .collect::<Vec<_>>();
    if completed.is_empty() {
        report.infeasible_contexts = 1;
        if first_failure_layer.is_none() {
            first_failure_layer = Some(context.steps.len().saturating_sub(1));
            failure_reason = Some(if final_duration_failures > 0 {
                "final_adjusted_duration_infeasible"
            } else {
                "final_plan_score_infeasible"
            });
        }
        append_context_report(
            &mut report,
            context,
            first_failure_layer,
            failure_reason,
            candidate_set_size_at_failure,
            domain_locally_feasible_candidates,
            retry_attempts,
            false,
        );
        if parameters.skip_infeasible {
            return Ok((OutputTable::default(), report));
        }
        return Err(SamplerError::NoFeasibleSequence {
            context_id: context.context_id,
            origin: context.initial_zone,
        });
    }
    report.completed_particles = completed.len() as u64;
    let max_weight = completed
        .iter()
        .map(|particle| particle.score - particle.log_proposal)
        .fold(f64::NEG_INFINITY, f64::max);
    let weights = completed
        .iter()
        .map(|particle| (particle.score - particle.log_proposal - max_weight).exp())
        .collect::<Vec<_>>();
    let weight_sum: f64 = weights.iter().sum();
    let squared_sum: f64 = weights.iter().map(|weight| weight * weight).sum();
    report.effective_sample_size_sum = weight_sum * weight_sum / squared_sum;
    completed.sort_unstable_by(|left, right| right.score.total_cmp(&left.score));
    let mut output = OutputTable::default();
    for (rank, particle) in completed
        .into_iter()
        .take(parameters.n_draws as usize)
        .enumerate()
    {
        append_plan(
            &mut output,
            graph,
            destinations,
            context,
            &problem,
            &particle.zones,
            parameters,
            rank as u32 + 1,
            particle.log_proposal,
            particle.score - particle.log_proposal,
        );
        report.selected_plans += 1;
    }
    if retry_attempts > 0 {
        report.recovered_contexts = 1;
    }
    append_context_report(
        &mut report,
        context,
        None,
        None,
        None,
        None,
        retry_attempts,
        retry_attempts > 0,
    );
    Ok((output, report))
}

pub fn sample_particles_all(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    contexts: &[Context],
    parameters: Parameters,
    particle_count: usize,
    candidate_count: usize,
    max_retries: u32,
    n_threads: Option<usize>,
) -> Result<(OutputTable, ParticleReport), SamplerError> {
    let compute = || {
        contexts
            .par_iter()
            .map(|context| {
                sample_context(
                    graph,
                    destinations,
                    context,
                    parameters,
                    particle_count,
                    candidate_count,
                    max_retries,
                )
            })
            .collect::<Vec<_>>()
    };
    let results = if let Some(n_threads) = n_threads {
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
    let mut report = ParticleReport::default();
    for result in results {
        match result {
            Ok((table, context_report)) => {
                output.extend(table);
                report.add(&context_report);
            }
            Err(error) => return Err(error),
        }
    }
    Ok((output, report))
}
