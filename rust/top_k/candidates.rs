//! Cached bounded candidate construction for bidirectional top-K search.

use std::collections::{BTreeSet, HashMap};

use crate::errors::SamplerError;
use crate::input::Context;
use crate::model::{DestinationIndex, OdGraph};

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
