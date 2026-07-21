//! Cached bounded candidate construction for bidirectional top-K search.

use std::collections::{BTreeSet, HashMap};

use crate::errors::SamplerError;
use crate::input::Context;
use crate::model::{DestinationIndex, OdGraph};
use crate::scoring::{score_local_weight, ScoringInputs};

#[derive(Default)]
pub(super) struct CandidateCache {
    base: HashMap<(usize, usize, bool), Vec<usize>>,
    reverse_projection: HashMap<(usize, usize), Vec<usize>>,
}

#[derive(Clone, Copy)]
pub(super) struct CandidateInputs<'a> {
    pub graph: &'a OdGraph,
    pub destinations: &'a DestinationIndex,
    pub context: &'a Context,
    pub scoring: ScoringInputs<'a>,
    pub candidate_count: usize,
    pub exploration_seed: u64,
}

#[derive(Clone, Copy)]
pub(super) struct CandidateQuery<'a> {
    pub layer: usize,
    pub reference_zone: usize,
    pub reverse: bool,
    pub state_index: usize,
    pub anchor_slot: Option<usize>,
    pub anchors: &'a [Option<usize>],
}

fn exploration_index(
    seed: u64,
    context_id: u64,
    state_index: usize,
    layer: usize,
    draw: u64,
    len: usize,
) -> usize {
    let mut value = seed
        ^ context_id.wrapping_mul(0x9E3779B97F4A7C15)
        ^ (state_index as u64).wrapping_mul(0xBF58476D1CE4E5B9)
        ^ (layer as u64).wrapping_mul(0x94D049BB133111EB)
        ^ draw.wrapping_mul(0xD6E8FEB86659FD93);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
    value ^= value >> 31;
    (value as usize) % len
}

fn base_candidates(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    context: &Context,
    layer: usize,
    reference_zone: usize,
    reverse: bool,
    candidate_count: usize,
) -> Result<Vec<usize>, SamplerError> {
    let step = context.steps[layer];
    if let Some(fixed) = step.fixed_destination {
        return Ok(vec![graph.zone_index[&fixed]]);
    }
    let values =
        destinations
            .activity(step.activity_id)
            .ok_or(SamplerError::NoFeasibleSequence {
                context_id: context.context_id,
                origin: context.initial_zone,
            })?;
    let mut result = BTreeSet::new();
    result.extend(
        destinations
            .attractive(step.activity_id)
            .unwrap_or(&[])
            .iter()
            .take(candidate_count)
            .copied(),
    );
    if reverse {
        result.extend(
            graph
                .incoming_by_cost(reference_zone)
                .filter(|(zone, _)| {
                    values[*zone].is_some_and(|value| value.log_opportunity_capacity.is_finite())
                })
                .take(candidate_count)
                .map(|(zone, _)| zone),
        );
    } else {
        result.extend(
            graph
                .outgoing_by_cost(reference_zone)
                .filter(|edge| {
                    values[edge.destination]
                        .is_some_and(|value| value.log_opportunity_capacity.is_finite())
                })
                .take(candidate_count)
                .map(|edge| edge.destination),
        );
    }
    Ok(result.into_iter().collect())
}

pub(super) fn candidates(
    inputs: CandidateInputs<'_>,
    query: CandidateQuery<'_>,
    cache: &mut CandidateCache,
) -> Result<Vec<usize>, SamplerError> {
    let step = inputs.context.steps[query.layer];
    if let Some(zone) = query.anchor_slot.and_then(|slot| query.anchors[slot]) {
        return Ok(vec![zone]);
    }
    let key = (query.layer, query.reference_zone, query.reverse);
    let base = if let Some(base) = cache.base.get(&key) {
        base
    } else {
        let base = base_candidates(
            inputs.graph,
            inputs.destinations,
            inputs.context,
            query.layer,
            query.reference_zone,
            query.reverse,
            inputs.candidate_count,
        )?;
        cache.base.entry(key).or_insert(base)
    };
    if step.fixed_destination.is_some() {
        return Ok(base.clone());
    }
    let domain =
        inputs
            .destinations
            .domain(step.activity_id)
            .ok_or(SamplerError::NoFeasibleSequence {
                context_id: inputs.context.context_id,
                origin: inputs.context.initial_zone,
            })?;
    let mut result = base.clone();
    for draw in 0..2 {
        let exploration = domain[exploration_index(
            inputs.exploration_seed,
            inputs.context.context_id,
            query.state_index,
            query.layer,
            draw,
            domain.len(),
        )];
        if let Err(index) = result.binary_search(&exploration) {
            result.insert(index, exploration);
        }
    }
    Ok(result)
}

/// Return the highest exact local-utility destinations for a known successor.
///
/// This deliberately scans the activity domain. It is the quality reference
/// for a future indexed utility surface: routing features are already compact
/// in `OdGraph`, while scoring remains owned by the shared scorer.
fn select_exact_local_candidates(
    inputs: CandidateInputs<'_>,
    query: CandidateQuery<'_>,
    next_zone: usize,
    candidate_zones: impl IntoIterator<Item = usize>,
) -> Result<Vec<usize>, SamplerError> {
    if let Some(zone) = query.anchor_slot.and_then(|slot| query.anchors[slot]) {
        return Ok(vec![zone]);
    }
    let step = inputs.context.steps[query.layer];
    if let Some(fixed) = step.fixed_destination {
        return Ok(vec![inputs.graph.zone_index[&fixed]]);
    }
    // Match the current 16 attractive + 16 travel-cost + 2 exploration
    // budget, so the first comparison changes ranking quality rather than the
    // candidate-pool width.
    let limit = inputs
        .candidate_count
        .saturating_mul(2)
        .saturating_add(2)
        .min(
            inputs
                .destinations
                .domain(step.activity_id)
                .map_or(0, <[usize]>::len),
        );
    let mut scored = candidate_zones
        .into_iter()
        .filter_map(|destination| {
            score_local_weight(
                inputs.scoring,
                query.layer,
                query.reference_zone,
                destination,
                Some(next_zone),
            )
            .map(|score| (score, destination))
        })
        .collect::<Vec<_>>();
    let compare = |left: &(f64, usize), right: &(f64, usize)| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
    };
    if scored.len() > limit {
        scored.select_nth_unstable_by(limit - 1, compare);
        scored.truncate(limit);
    }
    scored.sort_unstable_by(compare);
    Ok(scored
        .into_iter()
        .map(|(_, destination)| destination)
        .collect())
}

pub(super) fn exact_local_candidates(
    inputs: CandidateInputs<'_>,
    query: CandidateQuery<'_>,
    next_zone: usize,
) -> Result<Vec<usize>, SamplerError> {
    let domain = inputs
        .destinations
        .domain(inputs.context.steps[query.layer].activity_id)
        .ok_or(SamplerError::NoFeasibleSequence {
            context_id: inputs.context.context_id,
            origin: inputs.context.initial_zone,
        })?;
    select_exact_local_candidates(inputs, query, next_zone, domain.iter().copied())
}

pub(super) fn exact_local_candidates_from(
    inputs: CandidateInputs<'_>,
    query: CandidateQuery<'_>,
    next_zone: usize,
    candidate_zones: &[usize],
) -> Result<Vec<usize>, SamplerError> {
    select_exact_local_candidates(inputs, query, next_zone, candidate_zones.iter().copied())
}

/// Reverse counterpart of [`exact_local_candidates`].  The predecessor is
/// unknown on this front, but a candidate is the incoming origin of the known
/// next activity, so score that activity exactly instead.
pub(super) fn exact_reverse_local_candidates(
    inputs: CandidateInputs<'_>,
    query: CandidateQuery<'_>,
    next_zone: usize,
    next_next_zone: Option<usize>,
) -> Result<Vec<usize>, SamplerError> {
    if let Some(zone) = query.anchor_slot.and_then(|slot| query.anchors[slot]) {
        return Ok(vec![zone]);
    }
    let step = inputs.context.steps[query.layer];
    if let Some(fixed) = step.fixed_destination {
        return Ok(vec![inputs.graph.zone_index[&fixed]]);
    }
    let domain =
        inputs
            .destinations
            .domain(step.activity_id)
            .ok_or(SamplerError::NoFeasibleSequence {
                context_id: inputs.context.context_id,
                origin: inputs.context.initial_zone,
            })?;
    let limit = inputs
        .candidate_count
        .saturating_mul(2)
        .saturating_add(2)
        .min(domain.len());
    let mut scored = domain
        .iter()
        .filter_map(|&candidate| {
            score_local_weight(
                inputs.scoring,
                query.layer + 1,
                candidate,
                next_zone,
                next_next_zone,
            )
            .map(|score| (score, candidate))
        })
        .collect::<Vec<_>>();
    let compare = |left: &(f64, usize), right: &(f64, usize)| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
    };
    if scored.len() > limit {
        scored.select_nth_unstable_by(limit - 1, compare);
        scored.truncate(limit);
    }
    scored.sort_unstable_by(compare);
    Ok(scored.into_iter().map(|(_, candidate)| candidate).collect())
}

pub(super) fn reverse_projection_candidates(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    context: &Context,
    layer: usize,
    next_zone: usize,
    proposal_limit: usize,
    cache: &mut CandidateCache,
) -> Result<Vec<usize>, SamplerError> {
    if context.steps[layer].fixed_destination.is_some() {
        return Ok(Vec::new());
    }
    let key = (layer, next_zone);
    if let Some(candidates) = cache.reverse_projection.get(&key) {
        return Ok(candidates.clone());
    }
    let values = destinations
        .activity(context.steps[layer].activity_id)
        .ok_or(SamplerError::NoFeasibleSequence {
            context_id: context.context_id,
            origin: context.initial_zone,
        })?;
    let candidates = graph
        .incoming_by_cost(next_zone)
        .filter_map(|(zone, _)| {
            values[zone]
                .is_some_and(|value| value.log_opportunity_capacity.is_finite())
                .then_some(zone)
        })
        .take(proposal_limit)
        .collect::<Vec<_>>();
    cache.reverse_projection.insert(key, candidates.clone());
    Ok(candidates)
}
