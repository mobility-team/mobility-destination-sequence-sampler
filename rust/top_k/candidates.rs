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
    pub surface_bins: usize,
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

/// Diverse candidates from a precomputed utility surface.
///
/// The surface dimensions are attraction rank, inbound cost rank, and the
/// worse of inbound/outbound time-pressure ranks.  Keeping several winners
/// per cell prevents one speculative backward successor from collapsing the
/// forward beam into a single geographic region. The active resolution is
/// 2x2x2; the raw indexes also support experimental 4x4x4 maps.
pub(super) fn surface_candidates(
    inputs: CandidateInputs<'_>,
    query: CandidateQuery<'_>,
    next_zone: usize,
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
    let compare = |left: &(f64, usize), right: &(f64, usize)| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
    };
    let cell_count = inputs.surface_bins.pow(3);
    let per_cell = (inputs.candidate_count.saturating_mul(2))
        .div_ceil(cell_count)
        .max(1);
    let mut cells = vec![Vec::new(); cell_count];
    for &destination in domain {
        let Some(score) = score_local_weight(
            inputs.scoring,
            query.layer,
            query.reference_zone,
            destination,
            Some(next_zone),
        ) else {
            continue;
        };
        let Some(attraction_band) = inputs
            .destinations
            .attraction_band(step.activity_id, destination)
        else {
            continue;
        };
        let Some((cost_band, time_band)) =
            inputs
                .graph
                .surface_bands(query.reference_zone, destination, next_zone)
        else {
            continue;
        };
        let attraction_band = attraction_band as usize * inputs.surface_bins / 4;
        let cost_band = cost_band as usize * inputs.surface_bins / 4;
        let time_band = time_band as usize * inputs.surface_bins / 4;
        let cell =
            (attraction_band * inputs.surface_bins + cost_band) * inputs.surface_bins + time_band;
        let cell = &mut cells[cell];
        let rank =
            cell.partition_point(|existing| compare(existing, &(score, destination)).is_lt());
        if rank < per_cell {
            cell.insert(rank, (score, destination));
            cell.truncate(per_cell);
        }
    }
    let mut result = Vec::with_capacity(per_cell * cells.len());
    for cell in &cells {
        result.extend(cell.iter().copied());
    }
    result.sort_unstable_by(compare);
    result.truncate(inputs.candidate_count.saturating_mul(2));
    Ok(result
        .into_iter()
        .map(|(_, destination)| destination)
        .collect())
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
