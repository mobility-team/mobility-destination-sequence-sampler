//! Minimal bounded bidirectional top-K search.
//!
//! This is intentionally a bounded stitch-layer beam search, not the previous all-zone
//! bidirectional DP. It uses bounded candidate lists and beam frontiers in
//! both directions, scores every locally complete rigidity-aware factor on its
//! owning front, and scores only the two factors crossing the stitch boundary when
//! stitching.

use std::collections::{hash_map::Entry, BTreeSet, HashMap};
use std::time::Instant;

use rayon::prelude::*;

use crate::errors::SamplerError;
use crate::input::Context;
use crate::model::{DestinationIndex, OdGraph};
use crate::output::{OutputRow, OutputTable};
use crate::scoring::{
    adjusted_times, build_scoring_problem, fixed_destination_value, score_local_weight,
    score_local_weight_edges, score_local_weight_from_times, score_zones, Parameters,
    ScoringInputs, ScoringProblem,
};

mod candidates;

use candidates::{
    candidates, reverse_projection_candidates, surface_candidates, CandidateCache, CandidateInputs,
    CandidateQuery,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CandidateStrategy {
    Heuristic,
    Surface,
    FactorMap,
    SymmetricFactorMap,
}

impl CandidateStrategy {
    pub(crate) fn parse(value: &str) -> Result<Self, SamplerError> {
        match value {
            "heuristic" => Ok(Self::Heuristic),
            "surface" => Ok(Self::Surface),
            "factor_map" => Ok(Self::FactorMap),
            "symmetric_factor_map" => Ok(Self::SymmetricFactorMap),
            _ => Err(SamplerError::InvalidInput(
                "candidate_strategy must be 'surface', 'factor_map', 'symmetric_factor_map', or 'heuristic'"
                    .to_string(),
            )),
        }
    }
}

struct PrefixNode {
    parent: Option<usize>,
    zone: usize,
    exact_log_weight: f64,
    anchors: Vec<Option<usize>>,
}

struct SuffixNode {
    next: Option<usize>,
    zone: usize,
    exact_log_weight: f64,
    anchors: Vec<Option<usize>>,
}

struct BackwardMessages {
    nodes: Vec<SuffixNode>,
    /// Retained suffix boundary states by their current destination layer.
    /// A state at layer `i` owns factors `i + 1..`; evaluating a candidate at
    /// `i - 1` therefore needs the state at `i` and its `next` destination.
    frontiers: Vec<Vec<usize>>,
    /// A narrow continuation channel propagated from the stitch frontier back
    /// through the prefix layers. It guides proposal/ranking but never limits
    /// the wider frontier used for the exact final stitch.
    guidance_frontiers: Vec<Vec<usize>>,
    /// Independent right-to-left partial messages used only by the symmetric
    /// proposal channel. Exact suffix ownership remains in `nodes`.
    partial_frontiers: Vec<Vec<usize>>,
    /// Feasible reverse proposals for repeated anchors. These compact
    /// assignments preserve destination diversity without retaining a full
    /// suffix state for every anchor alternative.
    partial_anchor_candidates: Vec<Vec<usize>>,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum BackwardGuidanceMode {
    Exact,
    Partial,
}

#[derive(Default)]
struct LocalScoreCache {
    values: HashMap<(usize, usize, usize, Option<usize>), Option<f64>>,
}

#[derive(Default)]
struct FactorMapCache {
    /// Each exact map is indexed by the destination being proposed at the
    /// current layer.  The fixed neighbours in its key make it reusable across
    /// equivalent retained states without weakening rigidity feasibility.
    previous: HashMap<(usize, usize, usize), FactorScoreMap>,
    current: HashMap<(usize, usize, usize), FactorScoreMap>,
    next: HashMap<(usize, usize, Option<usize>), FactorScoreMap>,
    previous_hits: u64,
    previous_builds: u64,
    current_hits: u64,
    current_builds: u64,
    next_hits: u64,
    next_builds: u64,
}

/// Feasible factor scores, aligned to an activity-domain position rather than
/// the global zone index. Infeasible destinations are omitted completely.
#[derive(Default)]
struct FactorScoreMap {
    entries: Vec<(usize, f64)>,
}

impl LocalScoreCache {
    fn score(
        &mut self,
        inputs: ScoringInputs<'_>,
        layer: usize,
        origin: usize,
        destination: usize,
        next_destination: Option<usize>,
    ) -> Option<f64> {
        let key = (layer, origin, destination, next_destination);
        if let Some(score) = self.values.get(&key) {
            return *score;
        }
        let score = score_local_weight(inputs, layer, origin, destination, next_destination);
        self.values.insert(key, score);
        score
    }
}

fn anchor_slots(context: &Context) -> HashMap<u32, usize> {
    let mut slots = HashMap::new();
    for step in &context.steps {
        if let Some(anchor_id) = step.anchor_id {
            let next_slot = slots.len();
            slots.entry(anchor_id).or_insert(next_slot);
        }
    }
    slots
}

fn anchors_compatible(left: &[Option<usize>], right: &[Option<usize>]) -> bool {
    left.iter()
        .zip(right)
        .all(|(left, right)| left.is_none() || right.is_none() || left == right)
}

fn candidate_anchors_compatible(
    prefix_anchors: &[Option<usize>],
    suffix_anchors: &[Option<usize>],
    candidate_slot: Option<usize>,
    candidate: usize,
) -> bool {
    anchors_compatible(prefix_anchors, suffix_anchors)
        && candidate_slot
            .is_none_or(|slot| suffix_anchors[slot].is_none_or(|assigned| assigned == candidate))
}

#[derive(Debug, Default)]
pub struct TopKReport {
    pub contexts: u64,
    pub forward_candidate_evaluations: u64,
    pub backward_candidate_evaluations: u64,
    pub surface_proposal_evaluations: u64,
    pub factor_map_destination_evaluations: u64,
    pub factor_map_previous_hits: u64,
    pub factor_map_previous_builds: u64,
    pub factor_map_current_hits: u64,
    pub factor_map_current_builds: u64,
    pub factor_map_next_hits: u64,
    pub factor_map_next_builds: u64,
    pub continuation_proposals: u64,
    pub seam_refresh_proposals: u64,
    pub seam_refresh_states: u64,
    pub stitch_pairs: u64,
    pub completed_plans: u64,
    pub infeasible_contexts: u64,
    pub build_problem_ns: u64,
    pub backward_search_ns: u64,
    pub backward_guidance_ns: u64,
    pub forward_search_ns: u64,
    pub continuation_guidance_ns: u64,
    pub surface_proposal_ns: u64,
    pub factor_map_ns: u64,
    pub seam_refresh_ns: u64,
    pub stitch_ns: u64,
    pub materialize_ns: u64,
    pub total_search_ns: u64,
}

#[derive(Clone, Copy)]
pub struct TopKOptions {
    pub exploration_seed: u64,
    pub result_limit: u32,
    pub frontier_width: usize,
    pub proposal_limit_per_source: usize,
    pub symmetric_message_limit: usize,
    pub symmetric_state_limit: usize,
    pub symmetric_forward_proposal_limit: usize,
    pub candidate_strategy: CandidateStrategy,
    pub surface_bins: usize,
    pub factor_map_max_depth: usize,
    pub stitch_bias: i32,
    pub continuation_state_limit: usize,
    pub continuation_proposal_limit: usize,
    pub seam_refresh_per_prefix: usize,
    pub profile: bool,
}

/// Immutable data shared by every pass of one context search.
///
/// Keeping this separate from [`SearchScratch`] makes the dataflow between the
/// backward, guidance, forward, and stitch passes explicit.  It also keeps
/// pass signatures from growing whenever a new cache or report counter is
/// added.
struct SearchInputs<'a> {
    graph: &'a OdGraph,
    destinations: &'a DestinationIndex,
    context: &'a Context,
    problem: ScoringProblem,
    parameters: Parameters,
    options: TopKOptions,
    anchor_slots: HashMap<u32, usize>,
    repeated_anchor_slots: Vec<bool>,
}

impl SearchInputs<'_> {
    fn scoring(&self) -> ScoringInputs<'_> {
        ScoringInputs {
            graph: self.graph,
            destinations: self.destinations,
            context: self.context,
            problem: &self.problem,
            parameters: self.parameters,
        }
    }
}

/// Per-context mutable state used by the bounded search passes.
struct SearchScratch {
    candidate_cache: CandidateCache,
    factor_map_cache: FactorMapCache,
    factor_map_ranked: Vec<(f64, usize)>,
    local_scores: LocalScoreCache,
    report: TopKReport,
}

impl SearchScratch {
    fn new() -> Self {
        Self {
            candidate_cache: CandidateCache::default(),
            factor_map_cache: FactorMapCache::default(),
            factor_map_ranked: Vec::new(),
            local_scores: LocalScoreCache::default(),
            report: TopKReport {
                contexts: 1,
                ..TopKReport::default()
            },
        }
    }

    fn into_report(mut self) -> TopKReport {
        self.report.factor_map_previous_hits = self.factor_map_cache.previous_hits;
        self.report.factor_map_previous_builds = self.factor_map_cache.previous_builds;
        self.report.factor_map_current_hits = self.factor_map_cache.current_hits;
        self.report.factor_map_current_builds = self.factor_map_cache.current_builds;
        self.report.factor_map_next_hits = self.factor_map_cache.next_hits;
        self.report.factor_map_next_builds = self.factor_map_cache.next_builds;
        self.report
    }
}

impl TopKReport {
    fn add(&mut self, other: &Self) {
        self.contexts += other.contexts;
        self.forward_candidate_evaluations += other.forward_candidate_evaluations;
        self.backward_candidate_evaluations += other.backward_candidate_evaluations;
        self.surface_proposal_evaluations += other.surface_proposal_evaluations;
        self.factor_map_destination_evaluations += other.factor_map_destination_evaluations;
        self.factor_map_previous_hits += other.factor_map_previous_hits;
        self.factor_map_previous_builds += other.factor_map_previous_builds;
        self.factor_map_current_hits += other.factor_map_current_hits;
        self.factor_map_current_builds += other.factor_map_current_builds;
        self.factor_map_next_hits += other.factor_map_next_hits;
        self.factor_map_next_builds += other.factor_map_next_builds;
        self.continuation_proposals += other.continuation_proposals;
        self.seam_refresh_proposals += other.seam_refresh_proposals;
        self.seam_refresh_states += other.seam_refresh_states;
        self.stitch_pairs += other.stitch_pairs;
        self.completed_plans += other.completed_plans;
        self.infeasible_contexts += other.infeasible_contexts;
        self.build_problem_ns += other.build_problem_ns;
        self.backward_search_ns += other.backward_search_ns;
        self.backward_guidance_ns += other.backward_guidance_ns;
        self.forward_search_ns += other.forward_search_ns;
        self.continuation_guidance_ns += other.continuation_guidance_ns;
        self.surface_proposal_ns += other.surface_proposal_ns;
        self.factor_map_ns += other.factor_map_ns;
        self.seam_refresh_ns += other.seam_refresh_ns;
        self.stitch_ns += other.stitch_ns;
        self.materialize_ns += other.materialize_ns;
        self.total_search_ns += other.total_search_ns;
    }
}

fn select_beam_indices(scores: &[f64], beam_width: usize) -> Vec<usize> {
    let mut ranked = scores.iter().copied().enumerate().collect::<Vec<_>>();
    let compare = |(left_index, left): &(usize, f64), (right_index, right): &(usize, f64)| {
        right
            .total_cmp(left)
            .then_with(|| left_index.cmp(right_index))
    };
    if ranked.len() > beam_width {
        ranked.select_nth_unstable_by(beam_width - 1, compare);
        ranked.truncate(beam_width);
    }
    ranked.sort_unstable_by(compare);
    ranked
        .into_iter()
        .take(beam_width)
        .map(|(index, _)| index)
        .collect()
}

fn retain_pair_alternatives(
    scores: &[f64],
    pairs: &[(usize, usize)],
    per_pair_limit: usize,
) -> Vec<usize> {
    if per_pair_limit == 1 {
        return (0..scores.len()).collect();
    }
    let mut ranked = scores.iter().copied().enumerate().collect::<Vec<_>>();
    ranked.sort_unstable_by(|(left_index, left), (right_index, right)| {
        right
            .total_cmp(left)
            .then_with(|| left_index.cmp(right_index))
    });
    let mut retained = Vec::with_capacity(ranked.len());
    let mut per_pair = HashMap::<(usize, usize), usize>::new();
    for (index, _) in ranked {
        let count = per_pair.entry(pairs[index]).or_default();
        if *count < per_pair_limit {
            *count += 1;
            retained.push(index);
        }
    }
    retained
}

fn initial_endpoint_score(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    context: &Context,
    destination: usize,
    parameters: Parameters,
) -> Option<f64> {
    let step = context.steps[0];
    let origin = graph.zone_index[&context.initial_zone];
    let edge = graph.edge_to(origin, destination)?;
    let value = fixed_destination_value(destinations.activity(step.activity_id), destination);
    let attraction = if step.fixed_destination.is_some() {
        0.0
    } else if value.log_opportunity_capacity.is_finite() {
        value.log_opportunity_capacity
    } else {
        return None;
    };
    // The first activity has no outgoing leg yet, so its exact ternary factor
    // is deferred until the next forward extension. This is only a bounded
    // endpoint proposal, not a final activity score.
    Some(attraction - parameters.logit_scale * edge.cost)
}

struct ContinuationCandidate<'a> {
    layer: usize,
    previous_zone: usize,
    destination: usize,
    prefix_anchors: &'a [Option<usize>],
    anchor_slot: Option<usize>,
}

fn best_continuation_score(
    inputs: &SearchInputs<'_>,
    candidate: ContinuationCandidate<'_>,
    suffix_nodes: &[SuffixNode],
    suffix_frontier: &[usize],
    local_scores: &mut LocalScoreCache,
) -> Option<f64> {
    let mut best = f64::NEG_INFINITY;
    for &suffix_index in suffix_frontier {
        let suffix = &suffix_nodes[suffix_index];
        if !candidate_anchors_compatible(
            candidate.prefix_anchors,
            &suffix.anchors,
            candidate.anchor_slot,
            candidate.destination,
        ) {
            continue;
        }
        let score = local_scores
            .score(
                inputs.scoring(),
                candidate.layer,
                candidate.previous_zone,
                candidate.destination,
                Some(suffix.zone),
            )
            .and_then(|left| {
                local_scores
                    .score(
                        inputs.scoring(),
                        candidate.layer + 1,
                        candidate.destination,
                        suffix.zone,
                        suffix.next.map(|index| suffix_nodes[index].zone),
                    )
                    .map(|right| left + right + suffix.exact_log_weight)
            });
        if let Some(score) = score {
            best = best.max(score);
        }
    }
    best.is_finite().then_some(best)
}

struct FactorMapRequest<'a> {
    layer: usize,
    previous_zone: Option<usize>,
    origin: usize,
    suffixes: &'a [usize],
    anchor_slot: Option<usize>,
    anchors: &'a [Option<usize>],
    candidate_limit: usize,
}

struct NextFactorMapRequest<'a> {
    layer: usize,
    domain: &'a [usize],
    next_zone: usize,
    next_next_zone: Option<usize>,
}

fn next_factor_map<'a>(
    inputs: &SearchInputs<'_>,
    request: NextFactorMapRequest<'_>,
    cache: &'a mut HashMap<(usize, usize, Option<usize>), FactorScoreMap>,
    hits: &mut u64,
    builds: &mut u64,
) -> &'a FactorScoreMap {
    match cache.entry((request.layer + 1, request.next_zone, request.next_next_zone)) {
        Entry::Occupied(entry) => {
            *hits += 1;
            entry.into_mut()
        }
        Entry::Vacant(entry) => {
            *builds += 1;
            entry.insert({
                let next_outbound = request
                    .next_next_zone
                    .and_then(|zone| inputs.graph.edge_to(request.next_zone, zone));
                let next_departure = next_outbound.and_then(|edge| {
                    inputs
                        .context
                        .steps
                        .get(request.layer + 2)
                        .and_then(|step| {
                            adjusted_times(*step, edge).map(|(departure, _)| departure)
                        })
                });
                let mut map = FactorScoreMap {
                    entries: Vec::with_capacity(request.domain.len()),
                };
                for (position, &destination) in request.domain.iter().enumerate() {
                    let score = inputs
                        .graph
                        .edge_to(destination, request.next_zone)
                        .and_then(|inbound| {
                            let (_, arrival) =
                                adjusted_times(inputs.context.steps[request.layer + 1], inbound)?;
                            score_local_weight_from_times(
                                inputs.scoring(),
                                request.layer + 1,
                                request.next_zone,
                                inbound,
                                arrival,
                                next_departure,
                            )
                        });
                    if let Some(score) = score {
                        map.entries.push((position, score));
                    }
                }
                map
            })
        }
    }
}

struct ReverseFactorMapRequest<'a> {
    layer: usize,
    next_zone: usize,
    next_next_zone: Option<usize>,
    anchor_slot: Option<usize>,
    anchors: &'a [Option<usize>],
    candidate_limit: usize,
}

/// Known prefix utility components for a proposed reverse boundary. These are
/// recomputed from the boundary anchor assignment, never stored as exact
/// suffix utility: the first home leg and first-choice attractions are exact
/// components, while durations and unknown inbound legs remain deferred.
fn reverse_prefix_partial_score(
    inputs: &SearchInputs<'_>,
    layer: usize,
    destination: usize,
    anchors: &[Option<usize>],
    candidate_slot: Option<usize>,
) -> Option<f64> {
    let known_destination = |known_layer: usize| {
        if known_layer == layer {
            return Some(destination);
        }
        if let Some(fixed) = inputs.context.steps[known_layer].fixed_destination {
            return inputs.graph.zone_index.get(&fixed).copied();
        }
        inputs.context.steps[known_layer]
            .anchor_id
            .and_then(|anchor| inputs.anchor_slots.get(&anchor).copied())
            .and_then(|slot| {
                if candidate_slot == Some(slot) {
                    Some(destination)
                } else {
                    anchors[slot]
                }
            })
    };
    let mut score = 0.0;
    let mut exactly_scored = vec![false; layer + 1];
    for (factor_layer, exactly_scored) in exactly_scored.iter_mut().enumerate() {
        let Some(factor_destination) = known_destination(factor_layer) else {
            continue;
        };
        let factor_origin = if factor_layer == 0 {
            inputs.graph.zone_index[&inputs.context.initial_zone]
        } else {
            let Some(origin) = known_destination(factor_layer - 1) else {
                continue;
            };
            origin
        };
        let terminal_fixed = factor_layer + 1 == inputs.context.steps.len()
            && inputs.context.steps[factor_layer]
                .fixed_destination
                .is_some();
        let next_destination = if terminal_fixed {
            None
        } else {
            let Some(next) = known_destination(factor_layer + 1) else {
                continue;
            };
            Some(next)
        };
        score += score_local_weight(
            inputs.scoring(),
            factor_layer,
            factor_origin,
            factor_destination,
            next_destination,
        )?;
        *exactly_scored = true;
    }
    if !exactly_scored[0] {
        if let Some(first_destination) = known_destination(0) {
            score += initial_endpoint_score(
                inputs.graph,
                inputs.destinations,
                inputs.context,
                first_destination,
                inputs.parameters,
            )?;
        }
    }
    for (known_layer, &exactly_scored) in exactly_scored.iter().enumerate().skip(1) {
        if exactly_scored || !inputs.problem.is_first_choice(known_layer) {
            continue;
        }
        let Some(known_destination) = known_destination(known_layer) else {
            continue;
        };
        let step = inputs.context.steps[known_layer];
        let value = fixed_destination_value(
            inputs.destinations.activity(step.activity_id),
            known_destination,
        );
        if !value.log_opportunity_capacity.is_finite() {
            return None;
        }
        score += value.log_opportunity_capacity;
    }
    Some(score)
}

/// Rank a reverse extension by the exact factor already determined on its
/// right. The score is proposal-only; retained suffix nodes continue to own
/// and accumulate that exact factor through `LocalScoreCache`.
fn reverse_factor_map_candidates(
    inputs: &SearchInputs<'_>,
    request: ReverseFactorMapRequest<'_>,
    maps: &mut FactorMapCache,
    ranked: &mut Vec<(f64, usize)>,
) -> Result<Vec<usize>, SamplerError> {
    if let Some(zone) = request.anchor_slot.and_then(|slot| request.anchors[slot]) {
        return Ok(vec![zone]);
    }
    let step = inputs.context.steps[request.layer];
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
    let map = next_factor_map(
        inputs,
        NextFactorMapRequest {
            layer: request.layer,
            domain,
            next_zone: request.next_zone,
            next_next_zone: request.next_next_zone,
        },
        &mut maps.next,
        &mut maps.next_hits,
        &mut maps.next_builds,
    );
    ranked.clear();
    ranked.extend(map.entries.iter().filter_map(|&(position, score)| {
        let destination = domain[position];
        reverse_prefix_partial_score(
            inputs,
            request.layer,
            destination,
            request.anchors,
            request.anchor_slot,
        )
        .map(|partial| (score + partial, destination))
    }));
    let compare = |left: &(f64, usize), right: &(f64, usize)| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
    };
    if ranked.len() > request.candidate_limit {
        ranked.select_nth_unstable_by(request.candidate_limit - 1, compare);
        ranked.truncate(request.candidate_limit);
    }
    let mut result = ranked
        .iter()
        .map(|&(_, destination)| destination)
        .collect::<Vec<_>>();
    result.sort_unstable();
    Ok(result)
}

/// Build exact, destination-resolution utility maps for every activity factor
/// affected by choosing a forward destination. Missing entries are infeasible;
/// no sentinel values are introduced. This is an experimental alternative to
/// the binned surface: it ranks the intersection of the three maps directly.
fn factor_map_candidates(
    inputs: &SearchInputs<'_>,
    request: FactorMapRequest<'_>,
    suffix_nodes: &[SuffixNode],
    maps: &mut FactorMapCache,
    ranked: &mut Vec<(f64, usize)>,
) -> Result<Vec<usize>, SamplerError> {
    if let Some(zone) = request.anchor_slot.and_then(|slot| request.anchors[slot]) {
        return Ok(vec![zone]);
    }
    let step = inputs.context.steps[request.layer];
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
    let previous_map = request.previous_zone.map(|previous_zone| {
        let map = match maps
            .previous
            .entry((request.layer - 1, previous_zone, request.origin))
        {
            Entry::Occupied(entry) => {
                maps.previous_hits += 1;
                entry.into_mut()
            }
            Entry::Vacant(entry) => {
                maps.previous_builds += 1;
                entry.insert({
                    let inbound = inputs.graph.edge_to(previous_zone, request.origin);
                    let arrival = inbound.and_then(|edge| {
                        adjusted_times(inputs.context.steps[request.layer - 1], edge)
                            .map(|(_, arrival)| arrival)
                    });
                    let mut map = FactorScoreMap {
                        entries: Vec::with_capacity(domain.len()),
                    };
                    for (position, &destination) in domain.iter().enumerate() {
                        let score = inbound.and_then(|inbound| {
                            let outbound = inputs.graph.edge_to(request.origin, destination)?;
                            let next_departure = if inputs.parameters.update_plan_timings {
                                Some(
                                    adjusted_times(inputs.context.steps[request.layer], outbound)?
                                        .0,
                                )
                            } else {
                                None
                            };
                            score_local_weight_from_times(
                                inputs.scoring(),
                                request.layer - 1,
                                request.origin,
                                inbound,
                                arrival?,
                                next_departure,
                            )
                        });
                        if let Some(score) = score {
                            map.entries.push((position, score));
                        }
                    }
                    map
                })
            }
        };
        &*map
    });
    let compare = |left: &(f64, usize), right: &(f64, usize)| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
    };
    let mut result = Vec::with_capacity(request.candidate_limit * request.suffixes.len());
    for &suffix_index in request.suffixes {
        let suffix = &suffix_nodes[suffix_index];
        if let Some(assigned) = request.anchor_slot.and_then(|slot| suffix.anchors[slot]) {
            result.push(assigned);
            continue;
        }
        let next_zone = suffix.zone;
        let next_next_zone = suffix.next.map(|index| suffix_nodes[index].zone);
        let current_map =
            match maps
                .current
                .entry((request.layer, request.origin, next_zone))
            {
                Entry::Occupied(entry) => {
                    maps.current_hits += 1;
                    entry.into_mut()
                }
                Entry::Vacant(entry) => {
                    maps.current_builds += 1;
                    entry.insert({
                        let mut map = FactorScoreMap {
                            entries: Vec::with_capacity(domain.len()),
                        };
                        for (position, &destination) in domain.iter().enumerate() {
                            let score = inputs.graph.edge_to(request.origin, destination).and_then(
                                |inbound| {
                                    inputs.graph.edge_to(destination, next_zone).and_then(
                                        |outbound| {
                                            score_local_weight_edges(
                                                inputs.scoring(),
                                                request.layer,
                                                destination,
                                                inbound,
                                                Some(outbound),
                                            )
                                        },
                                    )
                                },
                            );
                            if let Some(score) = score {
                                map.entries.push((position, score));
                            }
                        }
                        map
                    })
                }
            };
        let next_map = next_factor_map(
            inputs,
            NextFactorMapRequest {
                layer: request.layer,
                domain,
                next_zone,
                next_next_zone,
            },
            &mut maps.next,
            &mut maps.next_hits,
            &mut maps.next_builds,
        );
        ranked.clear();
        let mut current_index = 0;
        let mut next_index = 0;
        let mut previous_index = 0;
        while current_index < current_map.entries.len() && next_index < next_map.entries.len() {
            let current_position = current_map.entries[current_index].0;
            let next_position = next_map.entries[next_index].0;
            let mut position = current_position.max(next_position);
            if let Some(previous_map) = previous_map {
                if previous_index == previous_map.entries.len() {
                    break;
                }
                position = position.max(previous_map.entries[previous_index].0);
            }
            let current = current_map.entries[current_index];
            let next = next_map.entries[next_index];
            let previous = previous_map.map(|map| map.entries[previous_index]);
            if current.0 != position
                || next.0 != position
                || previous.is_some_and(|value| value.0 != position)
            {
                if current.0 < position {
                    current_index += 1;
                }
                if next.0 < position {
                    next_index += 1;
                }
                if previous.is_some_and(|value| value.0 < position) {
                    previous_index += 1;
                }
                continue;
            }
            let previous_score = previous.map_or(0.0, |value| value.1);
            ranked.push((previous_score + current.1 + next.1, domain[current.0]));
            current_index += 1;
            next_index += 1;
            if previous_map.is_some() {
                previous_index += 1;
            }
        }
        if ranked.len() > request.candidate_limit {
            ranked.select_nth_unstable_by(request.candidate_limit - 1, compare);
            ranked.truncate(request.candidate_limit);
        }
        result.extend(ranked.iter().map(|&(_, destination)| destination));
    }
    result.sort_unstable();
    result.dedup();
    Ok(result)
}

fn forward_beam(
    inputs: &SearchInputs<'_>,
    scratch: &mut SearchScratch,
    stitch_layer: usize,
    backward: &BackwardMessages,
) -> Result<(Vec<PrefixNode>, Vec<usize>), SamplerError> {
    let graph = inputs.graph;
    let destinations = inputs.destinations;
    let context = inputs.context;
    let parameters = inputs.parameters;
    let beam_width = inputs.options.frontier_width;
    let candidate_count = inputs.options.proposal_limit_per_source;
    let anchor_slots = &inputs.anchor_slots;
    let continuation_state_limit = inputs.options.continuation_state_limit;
    let continuation_proposal_limit = inputs.options.continuation_proposal_limit;
    let factor_map_guidance_limit = continuation_state_limit.max(4);
    let symmetric = inputs.options.candidate_strategy == CandidateStrategy::SymmetricFactorMap;
    let symmetric_message_limit = inputs.options.symmetric_message_limit;
    let symmetric_state_limit = inputs.options.symmetric_state_limit;
    let partial_beam_reserve = symmetric_state_limit.min(beam_width);
    let profile = inputs.options.profile;
    let candidate_cache = &mut scratch.candidate_cache;
    let factor_map_cache = &mut scratch.factor_map_cache;
    let factor_map_ranked = &mut scratch.factor_map_ranked;
    let local_scores = &mut scratch.local_scores;
    let report = &mut scratch.report;
    let candidate_inputs = CandidateInputs {
        graph,
        destinations,
        context,
        scoring: inputs.scoring(),
        candidate_count,
        surface_bins: inputs.options.surface_bins,
        exploration_seed: inputs.options.exploration_seed,
    };
    let started = profile.then(Instant::now);
    let home = graph.zone_index[&context.initial_zone];
    let mut nodes = vec![PrefixNode {
        parent: None,
        zone: home,
        exact_log_weight: 0.0,
        anchors: vec![None; anchor_slots.len()],
    }];
    let mut frontier = vec![0];
    for layer in 0..=stitch_layer {
        let mut children = Vec::new();
        let mut scores = Vec::new();
        let mut partial_scores = Vec::new();
        let mut pairs = Vec::new();
        for (state_index, &parent_index) in frontier.iter().enumerate() {
            let parent = &nodes[parent_index];
            let candidate_slot = context.steps[layer]
                .anchor_id
                .and_then(|anchor| anchor_slots.get(&anchor).copied());
            let query = CandidateQuery {
                layer,
                reference_zone: parent.zone,
                reverse: false,
                state_index,
                anchor_slot: candidate_slot,
                anchors: &parent.anchors,
            };
            let guidance_suffixes = backward
                .guidance_frontiers
                .get(layer + 1)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let guidance_suffix = guidance_suffixes.first().copied();
            let unassigned = candidate_slot.is_none_or(|slot| parent.anchors[slot].is_none());
            let previous_zone = if layer == 0 {
                None
            } else if layer == 1 {
                Some(home)
            } else {
                Some(nodes[parent.parent.expect("non-root forward parent")].zone)
            };
            let mut candidate_zones = match (inputs.options.candidate_strategy, guidance_suffix) {
                (CandidateStrategy::Surface, Some(suffix_index)) if unassigned => {
                    let surface_started = profile.then(Instant::now);
                    if context.steps[layer].fixed_destination.is_none() {
                        report.surface_proposal_evaluations += destinations
                            .domain(context.steps[layer].activity_id)
                            .map_or(0, |domain| domain.len() as u64);
                    }
                    let result = surface_candidates(
                        candidate_inputs,
                        query,
                        backward.nodes[suffix_index].zone,
                    )?;
                    if let Some(started) = surface_started {
                        report.surface_proposal_ns += started.elapsed().as_nanos() as u64;
                    }
                    result
                }
                (strategy, Some(_))
                    if unassigned
                        && matches!(
                            strategy,
                            CandidateStrategy::FactorMap | CandidateStrategy::SymmetricFactorMap
                        ) =>
                {
                    let map_started = profile.then(Instant::now);
                    let factor_suffixes = if backward.frontiers[layer + 1].is_empty() {
                        guidance_suffixes
                    } else {
                        &backward.frontiers[layer + 1]
                    };
                    let map_count = factor_suffixes.len().min(factor_map_guidance_limit);
                    if context.steps[layer].fixed_destination.is_none() {
                        report.factor_map_destination_evaluations += destinations
                            .domain(context.steps[layer].activity_id)
                            .map_or(0, |domain| domain.len() as u64 * map_count as u64);
                    }
                    let per_map_limit = inputs
                        .options
                        .proposal_limit_per_source
                        .saturating_mul(2)
                        .div_ceil(map_count);
                    let mut result = Vec::with_capacity(per_map_limit * map_count);
                    let request = FactorMapRequest {
                        layer,
                        previous_zone,
                        origin: parent.zone,
                        suffixes: &factor_suffixes[..map_count],
                        anchor_slot: candidate_slot,
                        anchors: &parent.anchors,
                        candidate_limit: per_map_limit,
                    };
                    result.extend(factor_map_candidates(
                        inputs,
                        request,
                        &backward.nodes,
                        factor_map_cache,
                        factor_map_ranked,
                    )?);
                    if let Some(started) = map_started {
                        report.factor_map_ns += started.elapsed().as_nanos() as u64;
                    }
                    result
                }
                _ => candidates(candidate_inputs, query, candidate_cache)?,
            };
            if symmetric && unassigned {
                if let Some(slot) = candidate_slot {
                    if inputs.repeated_anchor_slots[slot] {
                        candidate_zones
                            .extend_from_slice(&backward.partial_anchor_candidates[slot]);
                    }
                }
            }
            if symmetric
                && unassigned
                && symmetric_message_limit > 0
                && inputs.options.symmetric_forward_proposal_limit > 0
            {
                let partial_suffixes = backward
                    .partial_frontiers
                    .get(layer + 1)
                    .map(Vec::as_slice)
                    .unwrap_or(&[]);
                let map_count = partial_suffixes.len().min(symmetric_message_limit);
                if map_count > 0 {
                    let map_started = profile.then(Instant::now);
                    if context.steps[layer].fixed_destination.is_none() {
                        report.factor_map_destination_evaluations += destinations
                            .domain(context.steps[layer].activity_id)
                            .map_or(0, |domain| domain.len() as u64 * map_count as u64);
                    }
                    let per_map_limit = inputs
                        .options
                        .symmetric_forward_proposal_limit
                        .div_ceil(map_count);
                    candidate_zones.extend(factor_map_candidates(
                        inputs,
                        FactorMapRequest {
                            layer,
                            previous_zone,
                            origin: parent.zone,
                            suffixes: &partial_suffixes[..map_count],
                            anchor_slot: candidate_slot,
                            anchors: &parent.anchors,
                            candidate_limit: per_map_limit,
                        },
                        &backward.nodes,
                        factor_map_cache,
                        factor_map_ranked,
                    )?);
                    if let Some(started) = map_started {
                        report.factor_map_ns += started.elapsed().as_nanos() as u64;
                    }
                }
            }
            let proposal_guidance_started = profile.then(Instant::now);
            if candidate_slot.is_none_or(|slot| parent.anchors[slot].is_none()) {
                if let Some(suffix_frontier) = backward.guidance_frontiers.get(layer + 1) {
                    for &suffix_index in suffix_frontier.iter().take(continuation_state_limit) {
                        let projections = reverse_projection_candidates(
                            graph,
                            destinations,
                            context,
                            layer,
                            backward.nodes[suffix_index].zone,
                            continuation_proposal_limit,
                            candidate_cache,
                        )?;
                        report.continuation_proposals += projections.len() as u64;
                        candidate_zones.extend(projections);
                    }
                }
            }
            candidate_zones.sort_unstable();
            candidate_zones.dedup();
            if let Some(started) = proposal_guidance_started {
                report.continuation_guidance_ns += started.elapsed().as_nanos() as u64;
            }
            for candidate in candidate_zones {
                report.forward_candidate_evaluations += 1;
                let local_score = if layer == 0 {
                    initial_endpoint_score(graph, destinations, context, candidate, parameters)
                } else {
                    local_scores.score(
                        inputs.scoring(),
                        layer - 1,
                        previous_zone.expect("non-initial layer has a previous destination"),
                        parent.zone,
                        Some(candidate),
                    )
                };
                if let Some(local_score) = local_score {
                    let prefix_score = if layer == 0 {
                        0.0
                    } else {
                        parent.exact_log_weight + local_score
                    };
                    let continuation_started = profile.then(Instant::now);
                    let continuation_score = backward
                        .guidance_frontiers
                        .get(layer + 1)
                        .filter(|_| continuation_state_limit > 0)
                        .and_then(|suffix_frontier| {
                            best_continuation_score(
                                inputs,
                                ContinuationCandidate {
                                    layer,
                                    previous_zone: parent.zone,
                                    destination: candidate,
                                    prefix_anchors: &parent.anchors,
                                    anchor_slot: candidate_slot,
                                },
                                &backward.nodes,
                                &suffix_frontier
                                    [..suffix_frontier.len().min(continuation_state_limit)],
                                local_scores,
                            )
                        });
                    let primary_score = continuation_score
                        .map(|score| prefix_score + score)
                        .unwrap_or_else(|| {
                            if layer == 0 {
                                local_score
                            } else {
                                prefix_score
                            }
                        });
                    let partial_score = if symmetric {
                        backward
                            .partial_frontiers
                            .get(layer + 1)
                            .filter(|frontier| symmetric_message_limit > 0 && !frontier.is_empty())
                            .and_then(|suffix_frontier| {
                                best_continuation_score(
                                    inputs,
                                    ContinuationCandidate {
                                        layer,
                                        previous_zone: parent.zone,
                                        destination: candidate,
                                        prefix_anchors: &parent.anchors,
                                        anchor_slot: candidate_slot,
                                    },
                                    &backward.nodes,
                                    &suffix_frontier
                                        [..suffix_frontier.len().min(symmetric_message_limit)],
                                    local_scores,
                                )
                            })
                            .map(|score| prefix_score + score)
                            .unwrap_or(primary_score)
                    } else {
                        primary_score
                    };
                    if let Some(started) = continuation_started {
                        report.continuation_guidance_ns += started.elapsed().as_nanos() as u64;
                    }
                    children.push((
                        parent_index,
                        candidate,
                        if layer == 0 { 0.0 } else { local_score },
                    ));
                    pairs.push((parent.zone, candidate));
                    scores.push(primary_score);
                    partial_scores.push(partial_score);
                }
            }
        }
        if children.is_empty() {
            frontier.clear();
            break;
        }
        let mut retained = retain_pair_alternatives(
            &scores,
            &pairs,
            (inputs.options.result_limit as usize).min(beam_width),
        );
        if symmetric && partial_beam_reserve > 0 {
            for index in retain_pair_alternatives(
                &partial_scores,
                &pairs,
                (inputs.options.result_limit as usize).min(partial_beam_reserve),
            ) {
                if !retained.contains(&index) {
                    retained.push(index);
                }
            }
        }
        let children = retained
            .iter()
            .map(|&index| children[index])
            .collect::<Vec<_>>();
        let scores = retained
            .iter()
            .map(|&index| scores[index])
            .collect::<Vec<_>>();
        let partial_scores = retained
            .iter()
            .map(|&index| partial_scores[index])
            .collect::<Vec<_>>();
        let mut selected = select_beam_indices(&scores, beam_width);
        if symmetric && partial_beam_reserve > 0 {
            for index in select_beam_indices(&partial_scores, partial_beam_reserve) {
                if !selected.contains(&index) {
                    selected.push(index);
                }
            }
        }
        frontier = selected
            .into_iter()
            .map(|index| {
                let (parent_index, destination, exact_increment) = children[index];
                let node_index = nodes.len();
                let mut anchors = nodes[parent_index].anchors.clone();
                if let Some(anchor) = context.steps[layer].anchor_id {
                    anchors[anchor_slots[&anchor]] = Some(destination);
                }
                nodes.push(PrefixNode {
                    parent: Some(parent_index),
                    zone: destination,
                    exact_log_weight: nodes[parent_index].exact_log_weight + exact_increment,
                    anchors,
                });
                node_index
            })
            .collect();
    }
    if let Some(started) = started {
        report.forward_search_ns += started.elapsed().as_nanos() as u64;
    }
    Ok((nodes, frontier))
}

/// Add a small set of suffix boundary states proposed from the retained forward
/// frontier. The original backward frontier remains intact: this is a bounded
/// F-to-B seam refresh, not a replacement of reverse candidate generation.
struct RefreshSuffixRequest<'a> {
    prefix_anchors: &'a [Option<usize>],
    candidate_slot: Option<usize>,
    candidate: usize,
    refresh_layer: usize,
    downstream: &'a [usize],
    messages: &'a BackwardMessages,
}

fn best_refresh_suffix(
    inputs: &SearchInputs<'_>,
    local_scores: &mut LocalScoreCache,
    request: RefreshSuffixRequest<'_>,
    unanchored_values: &mut HashMap<usize, Option<(usize, f64, f64)>>,
) -> Option<(usize, f64, f64)> {
    if inputs.anchor_slots.is_empty() {
        if let Some(value) = unanchored_values.get(&request.candidate) {
            return *value;
        }
    }
    let mut best = None;
    for &next_index in request.downstream {
        let next = &request.messages.nodes[next_index];
        if !inputs.anchor_slots.is_empty()
            && !candidate_anchors_compatible(
                request.prefix_anchors,
                &next.anchors,
                request.candidate_slot,
                request.candidate,
            )
        {
            continue;
        }
        let Some(local_score) = local_scores.score(
            inputs.scoring(),
            request.refresh_layer + 1,
            request.candidate,
            next.zone,
            next.next.map(|index| request.messages.nodes[index].zone),
        ) else {
            continue;
        };
        let suffix_score = next.exact_log_weight + local_score;
        if best.is_none_or(|(_, _, score)| suffix_score > score) {
            best = Some((next_index, local_score, suffix_score));
        }
    }
    if inputs.anchor_slots.is_empty() {
        unanchored_values.insert(request.candidate, best);
    }
    best
}

fn refresh_stitch_frontier(
    inputs: &SearchInputs<'_>,
    scratch: &mut SearchScratch,
    stitch_layer: usize,
    prefix_nodes: &[PrefixNode],
    prefix_frontier: &[usize],
    messages: &mut BackwardMessages,
) -> Result<(), SamplerError> {
    let graph = inputs.graph;
    let destinations = inputs.destinations;
    let context = inputs.context;
    let candidate_count = inputs.options.proposal_limit_per_source;
    let refresh_per_prefix = inputs.options.seam_refresh_per_prefix;
    let anchor_slots = &inputs.anchor_slots;
    let profile = inputs.options.profile;
    let candidate_cache = &mut scratch.candidate_cache;
    let local_scores = &mut scratch.local_scores;
    let report = &mut scratch.report;
    let candidate_inputs = CandidateInputs {
        graph,
        destinations,
        context,
        scoring: inputs.scoring(),
        candidate_count,
        surface_bins: inputs.options.surface_bins,
        exploration_seed: inputs.options.exploration_seed,
    };
    if refresh_per_prefix == 0 {
        return Ok(());
    }
    let refresh_layer = stitch_layer + 1;
    if refresh_layer + 1 >= context.steps.len() {
        // The stitch suffix is the fixed terminal destination, so there is no
        // activity destination to refresh.
        return Ok(());
    }
    let started = profile.then(Instant::now);
    let home = graph.zone_index[&context.initial_zone];
    let downstream = messages.frontiers[refresh_layer + 1].clone();
    if downstream.is_empty() {
        return Ok(());
    }

    let mut existing = messages.frontiers[refresh_layer]
        .iter()
        .map(|&index| {
            let node = &messages.nodes[index];
            (node.zone, node.next.expect("non-terminal stitch suffix"))
        })
        .collect::<BTreeSet<_>>();
    let mut additions = Vec::new();
    let mut unanchored_suffix_values = HashMap::<usize, Option<(usize, f64, f64)>>::new();
    for (state_index, &prefix_index) in prefix_frontier.iter().enumerate() {
        let prefix = &prefix_nodes[prefix_index];
        let candidate_slot = context.steps[refresh_layer]
            .anchor_id
            .and_then(|anchor| anchor_slots.get(&anchor).copied());
        let mut ranked = Vec::new();
        for candidate in candidates(
            candidate_inputs,
            CandidateQuery {
                layer: refresh_layer,
                reference_zone: prefix.zone,
                reverse: false,
                state_index,
                anchor_slot: candidate_slot,
                anchors: &prefix.anchors,
            },
            candidate_cache,
        )? {
            report.seam_refresh_proposals += 1;
            let best = best_refresh_suffix(
                inputs,
                local_scores,
                RefreshSuffixRequest {
                    prefix_anchors: &prefix.anchors,
                    candidate_slot,
                    candidate,
                    refresh_layer,
                    downstream: &downstream,
                    messages,
                },
                &mut unanchored_suffix_values,
            );
            let Some((next_index, local_score, suffix_score)) = best else {
                continue;
            };
            let prefix_previous = if stitch_layer == 0 {
                home
            } else {
                prefix_nodes[prefix.parent.expect("non-root stitch prefix")].zone
            };
            let Some(boundary_score) = local_scores.score(
                inputs.scoring(),
                stitch_layer,
                prefix_previous,
                prefix.zone,
                Some(candidate),
            ) else {
                continue;
            };
            ranked.push((
                prefix.exact_log_weight + boundary_score + suffix_score,
                candidate,
                next_index,
                local_score,
            ));
        }
        let compare = |left: &(f64, usize, usize, f64), right: &(f64, usize, usize, f64)| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.1.cmp(&right.1))
                .then_with(|| left.2.cmp(&right.2))
        };
        if ranked.len() > refresh_per_prefix {
            ranked.select_nth_unstable_by(refresh_per_prefix - 1, compare);
            ranked.truncate(refresh_per_prefix);
        }
        ranked.sort_unstable_by(compare);
        for (_, candidate, next_index, local_score) in ranked {
            if !existing.insert((candidate, next_index)) {
                continue;
            }
            additions.push((candidate, next_index, local_score));
        }
    }
    for (candidate, next_index, local_score) in additions {
        let node_index = messages.nodes.len();
        let mut anchors = messages.nodes[next_index].anchors.clone();
        if let Some(anchor) = context.steps[refresh_layer].anchor_id {
            anchors[anchor_slots[&anchor]] = Some(candidate);
        }
        messages.nodes.push(SuffixNode {
            next: Some(next_index),
            zone: candidate,
            exact_log_weight: messages.nodes[next_index].exact_log_weight + local_score,
            anchors,
        });
        messages.frontiers[refresh_layer].push(node_index);
        report.seam_refresh_states += 1;
    }
    if let Some(started) = started {
        report.seam_refresh_ns += started.elapsed().as_nanos() as u64;
    }
    Ok(())
}

fn backward_beam(
    inputs: &SearchInputs<'_>,
    scratch: &mut SearchScratch,
    first_layer: usize,
) -> Result<BackwardMessages, SamplerError> {
    let graph = inputs.graph;
    let destinations = inputs.destinations;
    let context = inputs.context;
    let beam_width = inputs.options.frontier_width;
    let candidate_count = inputs.options.proposal_limit_per_source;
    let anchor_slots = &inputs.anchor_slots;
    let profile = inputs.options.profile;
    let candidate_cache = &mut scratch.candidate_cache;
    let local_scores = &mut scratch.local_scores;
    let report = &mut scratch.report;
    let candidate_inputs = CandidateInputs {
        graph,
        destinations,
        context,
        scoring: inputs.scoring(),
        candidate_count,
        surface_bins: inputs.options.surface_bins,
        exploration_seed: inputs.options.exploration_seed,
    };
    let started = profile.then(Instant::now);
    let terminal = context.steps.last().expect("context has steps");
    let terminal_zone = terminal.fixed_destination.ok_or_else(|| {
        SamplerError::InvalidInput(format!(
            "context {} needs a fixed terminal destination for bidirectional top-K search",
            context.context_id
        ))
    })?;
    let mut nodes = vec![SuffixNode {
        next: None,
        zone: graph.zone_index[&terminal_zone],
        exact_log_weight: 0.0,
        anchors: vec![None; anchor_slots.len()],
    }];
    let mut frontier = vec![0];
    let terminal_layer = context.steps.len() - 1;
    let mut frontiers = vec![Vec::new(); context.steps.len()];
    frontiers[terminal_layer] = frontier.clone();
    for layer in (first_layer..terminal_layer).rev() {
        let mut children = Vec::new();
        let mut scores = Vec::new();
        let mut pairs = Vec::new();
        for (state_index, &next_index) in frontier.iter().enumerate() {
            let next = &nodes[next_index];
            let next_zone = next.zone;
            let query = CandidateQuery {
                layer,
                reference_zone: next_zone,
                reverse: true,
                state_index,
                anchor_slot: context.steps[layer]
                    .anchor_id
                    .and_then(|anchor| anchor_slots.get(&anchor).copied()),
                anchors: &next.anchors,
            };
            let next_next_zone = next.next.map(|index| nodes[index].zone);
            let reverse_candidates = candidates(candidate_inputs, query, candidate_cache)?;
            for candidate in reverse_candidates {
                report.backward_candidate_evaluations += 1;
                let score = local_scores.score(
                    inputs.scoring(),
                    layer + 1,
                    candidate,
                    next_zone,
                    next_next_zone,
                );
                if let Some(score) = score {
                    children.push((next_index, candidate, score));
                    pairs.push((candidate, next_zone));
                    scores.push(next.exact_log_weight + score);
                }
            }
        }
        if children.is_empty() {
            frontier.clear();
            break;
        }
        let retained = retain_pair_alternatives(
            &scores,
            &pairs,
            (inputs.options.result_limit as usize).min(beam_width),
        );
        let children = retained
            .iter()
            .map(|&index| children[index])
            .collect::<Vec<_>>();
        let scores = retained
            .iter()
            .map(|&index| scores[index])
            .collect::<Vec<_>>();
        frontier = select_beam_indices(&scores, beam_width)
            .into_iter()
            .map(|index| {
                let (next_index, destination, exact_increment) = children[index];
                let node_index = nodes.len();
                let mut anchors = nodes[next_index].anchors.clone();
                if let Some(anchor) = context.steps[layer].anchor_id {
                    anchors[anchor_slots[&anchor]] = Some(destination);
                }
                nodes.push(SuffixNode {
                    next: Some(next_index),
                    zone: destination,
                    exact_log_weight: nodes[next_index].exact_log_weight + exact_increment,
                    anchors,
                });
                node_index
            })
            .collect();
        frontiers[layer] = frontier.clone();
    }
    if let Some(started) = started {
        report.backward_search_ns += started.elapsed().as_nanos() as u64;
    }
    Ok(BackwardMessages {
        nodes,
        frontiers,
        guidance_frontiers: vec![Vec::new(); context.steps.len()],
        partial_frontiers: vec![Vec::new(); context.steps.len()],
        partial_anchor_candidates: vec![Vec::new(); anchor_slots.len()],
    })
}

fn extend_backward_guidance(
    inputs: &SearchInputs<'_>,
    scratch: &mut SearchScratch,
    stitch_layer: usize,
    messages: &mut BackwardMessages,
    mode: BackwardGuidanceMode,
) -> Result<(), SamplerError> {
    let graph = inputs.graph;
    let destinations = inputs.destinations;
    let context = inputs.context;
    let guidance_width = match mode {
        BackwardGuidanceMode::Partial => inputs.options.symmetric_message_limit,
        BackwardGuidanceMode::Exact
            if matches!(
                inputs.options.candidate_strategy,
                CandidateStrategy::FactorMap | CandidateStrategy::SymmetricFactorMap
            ) =>
        {
            inputs.options.continuation_state_limit.max(4)
        }
        BackwardGuidanceMode::Exact => inputs.options.continuation_state_limit,
    };
    let candidate_count = inputs.options.proposal_limit_per_source;
    let anchor_slots = &inputs.anchor_slots;
    let profile = inputs.options.profile;
    let candidate_cache = &mut scratch.candidate_cache;
    let factor_map_cache = &mut scratch.factor_map_cache;
    let factor_map_ranked = &mut scratch.factor_map_ranked;
    let local_scores = &mut scratch.local_scores;
    let report = &mut scratch.report;
    let candidate_inputs = CandidateInputs {
        graph,
        destinations,
        context,
        scoring: inputs.scoring(),
        candidate_count,
        surface_bins: inputs.options.surface_bins,
        exploration_seed: inputs.options.exploration_seed,
    };
    let started = profile.then(Instant::now);
    let mut frontier = messages.frontiers[stitch_layer + 1]
        .iter()
        .copied()
        .take(guidance_width)
        .collect::<Vec<_>>();
    match mode {
        BackwardGuidanceMode::Exact => {
            messages.guidance_frontiers[stitch_layer + 1] = frontier.clone()
        }
        BackwardGuidanceMode::Partial => {
            messages.partial_frontiers[stitch_layer + 1] = frontier.clone()
        }
    }
    for layer in (0..=stitch_layer).rev() {
        let layer_width = if mode == BackwardGuidanceMode::Partial && layer < stitch_layer {
            inputs.options.symmetric_state_limit
        } else {
            guidance_width
        };
        let mut children = Vec::new();
        let mut scores = Vec::new();
        let mut pairs = Vec::new();
        for (state_index, &next_index) in frontier.iter().enumerate() {
            let next = &messages.nodes[next_index];
            let next_zone = next.zone;
            let query = CandidateQuery {
                layer,
                reference_zone: next_zone,
                reverse: true,
                state_index,
                anchor_slot: context.steps[layer]
                    .anchor_id
                    .and_then(|anchor| anchor_slots.get(&anchor).copied()),
                anchors: &next.anchors,
            };
            let next_next_zone = next.next.map(|index| messages.nodes[index].zone);
            let reverse_candidates = if mode == BackwardGuidanceMode::Partial {
                let map_started = profile.then(Instant::now);
                if context.steps[layer].fixed_destination.is_none()
                    && query
                        .anchor_slot
                        .is_none_or(|slot| query.anchors[slot].is_none())
                {
                    report.factor_map_destination_evaluations += destinations
                        .domain(context.steps[layer].activity_id)
                        .map_or(0, |domain| domain.len() as u64);
                }
                let result = reverse_factor_map_candidates(
                    inputs,
                    ReverseFactorMapRequest {
                        layer,
                        next_zone,
                        next_next_zone,
                        anchor_slot: query.anchor_slot,
                        anchors: query.anchors,
                        candidate_limit: candidate_count.saturating_mul(2),
                    },
                    factor_map_cache,
                    factor_map_ranked,
                )?;
                if let Some(started) = map_started {
                    report.factor_map_ns += started.elapsed().as_nanos() as u64;
                }
                result
            } else {
                candidates(candidate_inputs, query, candidate_cache)?
            };
            for candidate in reverse_candidates {
                report.backward_candidate_evaluations += 1;
                let score = local_scores.score(
                    inputs.scoring(),
                    layer + 1,
                    candidate,
                    next_zone,
                    next_next_zone,
                );
                if let Some(score) = score {
                    let partial = if mode == BackwardGuidanceMode::Partial {
                        let Some(partial) = reverse_prefix_partial_score(
                            inputs,
                            layer,
                            candidate,
                            query.anchors,
                            query.anchor_slot,
                        ) else {
                            continue;
                        };
                        partial
                    } else {
                        0.0
                    };
                    children.push((next_index, candidate, score));
                    pairs.push((candidate, next_zone));
                    scores.push(next.exact_log_weight + score + partial);
                }
            }
        }
        if children.is_empty() {
            frontier.clear();
            break;
        }
        if mode == BackwardGuidanceMode::Partial {
            if let Some(anchor) = context.steps[layer].anchor_id {
                let slot = anchor_slots[&anchor];
                if inputs.repeated_anchor_slots[slot] {
                    let compact = &mut messages.partial_anchor_candidates[slot];
                    for index in select_beam_indices(
                        &scores,
                        inputs.options.symmetric_forward_proposal_limit,
                    ) {
                        let (next_index, destination, _) = children[index];
                        if messages.nodes[next_index].anchors[slot].is_none()
                            && !compact.contains(&destination)
                        {
                            compact.push(destination);
                        }
                    }
                }
            }
        }
        let retained = retain_pair_alternatives(
            &scores,
            &pairs,
            (inputs.options.result_limit as usize).min(layer_width),
        );
        let children = retained
            .iter()
            .map(|&index| children[index])
            .collect::<Vec<_>>();
        let scores = retained
            .iter()
            .map(|&index| scores[index])
            .collect::<Vec<_>>();
        frontier = select_beam_indices(&scores, layer_width)
            .into_iter()
            .map(|index| {
                let (next_index, destination, exact_increment) = children[index];
                let node_index = messages.nodes.len();
                let mut anchors = messages.nodes[next_index].anchors.clone();
                if let Some(anchor) = context.steps[layer].anchor_id {
                    anchors[anchor_slots[&anchor]] = Some(destination);
                }
                messages.nodes.push(SuffixNode {
                    next: Some(next_index),
                    zone: destination,
                    exact_log_weight: messages.nodes[next_index].exact_log_weight + exact_increment,
                    anchors,
                });
                node_index
            })
            .collect();
        match mode {
            BackwardGuidanceMode::Exact => messages.guidance_frontiers[layer] = frontier.clone(),
            BackwardGuidanceMode::Partial => messages.partial_frontiers[layer] = frontier.clone(),
        }
    }
    if let Some(started) = started {
        report.backward_guidance_ns += started.elapsed().as_nanos() as u64;
    }
    Ok(())
}

fn prefix_zones(nodes: &[PrefixNode], mut index: usize) -> Vec<usize> {
    let mut zones = Vec::new();
    while let Some(parent) = nodes[index].parent {
        zones.push(nodes[index].zone);
        index = parent;
    }
    zones.reverse();
    zones
}

fn suffix_zones(nodes: &[SuffixNode], mut index: usize) -> Vec<usize> {
    let mut zones = Vec::new();
    loop {
        zones.push(nodes[index].zone);
        let Some(next) = nodes[index].next else {
            return zones;
        };
        index = next;
    }
}

struct CompletedPlan {
    score: f64,
    prefix: usize,
    suffix: usize,
}

fn append_plan(output: &mut OutputTable, inputs: &SearchInputs<'_>, zones: &[usize], draw_id: u32) {
    let Some((_, local_weights)) = score_zones(inputs.scoring(), zones) else {
        return;
    };
    let mut suffixes = vec![0.0; zones.len()];
    let mut suffix = 0.0;
    for layer in (0..zones.len()).rev() {
        suffix += local_weights[layer];
        suffixes[layer] = suffix;
    }
    let mut origin = inputs.graph.zone_index[&inputs.context.initial_zone];
    for (layer, &destination) in zones.iter().enumerate() {
        output.push(OutputRow {
            context_id: inputs.context.context_id,
            draw_id,
            layer: layer as u32,
            origin: inputs.graph.zone_ids[origin],
            destination: inputs.graph.zone_ids[destination],
            local_log_weight: local_weights[layer],
            total_log_weight: suffixes[layer],
        });
        origin = destination;
    }
}

fn search_two_step_context(
    inputs: &SearchInputs<'_>,
    scratch: &mut SearchScratch,
) -> Result<OutputTable, SamplerError> {
    let search_started = inputs.options.profile.then(Instant::now);
    let terminal = inputs.context.steps[1];
    let terminal_zone = terminal.fixed_destination.ok_or_else(|| {
        SamplerError::InvalidInput(format!(
            "context {} needs a fixed terminal destination for top-K search",
            inputs.context.context_id
        ))
    })?;
    let terminal = *inputs.graph.zone_index.get(&terminal_zone).ok_or_else(|| {
        SamplerError::InvalidInput(format!(
            "context {} terminal destination {} is absent from the OD graph",
            inputs.context.context_id, terminal_zone
        ))
    })?;
    let first = inputs.context.steps[0];
    let candidates = if let Some(fixed) = first.fixed_destination {
        vec![*inputs.graph.zone_index.get(&fixed).ok_or_else(|| {
            SamplerError::InvalidInput(format!(
                "context {} fixed destination {} is absent from the OD graph",
                inputs.context.context_id, fixed
            ))
        })?]
    } else {
        inputs
            .destinations
            .domain(first.activity_id)
            .ok_or(SamplerError::NoFeasibleSequence {
                context_id: inputs.context.context_id,
                origin: inputs.context.initial_zone,
            })?
            .to_vec()
    };
    let mut completed = Vec::with_capacity(candidates.len());
    for destination in candidates {
        scratch.report.forward_candidate_evaluations += 1;
        let zones = vec![destination, terminal];
        if let Some((score, _)) = score_zones(inputs.scoring(), &zones) {
            completed.push((score, zones));
        }
    }
    if let Some(started) = search_started {
        scratch.report.forward_search_ns += started.elapsed().as_nanos() as u64;
    }
    if completed.is_empty() {
        scratch.report.infeasible_contexts = 1;
        return Err(SamplerError::NoFeasibleSequence {
            context_id: inputs.context.context_id,
            origin: inputs.context.initial_zone,
        });
    }
    completed.sort_unstable_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    scratch.report.completed_plans = completed.len() as u64;
    let materialize_started = inputs.options.profile.then(Instant::now);
    let mut output = OutputTable::default();
    for (draw, (_, zones)) in completed
        .into_iter()
        .take(inputs.options.result_limit as usize)
        .enumerate()
    {
        append_plan(&mut output, inputs, &zones, draw as u32 + 1);
    }
    if let Some(started) = materialize_started {
        scratch.report.materialize_ns += started.elapsed().as_nanos() as u64;
    }
    Ok(output)
}

fn search_context(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    context: &Context,
    parameters: Parameters,
    options: TopKOptions,
) -> Result<(OutputTable, TopKReport), SamplerError> {
    let started = options.profile.then(Instant::now);
    if context.steps.len() < 2 {
        return Err(SamplerError::InvalidInput(format!(
            "context {} needs at least two steps for top-K search",
            context.context_id
        )));
    }
    let build_started = options.profile.then(Instant::now);
    let problem = build_scoring_problem(context)?;
    let use_heuristic = match options.candidate_strategy {
        CandidateStrategy::Surface => context.steps.len() > 4,
        CandidateStrategy::FactorMap | CandidateStrategy::SymmetricFactorMap => {
            context.steps.len() > options.factor_map_max_depth
        }
        CandidateStrategy::Heuristic => false,
    };
    let options = if use_heuristic {
        TopKOptions {
            candidate_strategy: CandidateStrategy::Heuristic,
            ..options
        }
    } else {
        options
    };
    let anchor_slots = anchor_slots(context);
    let mut anchor_counts = vec![0_u32; anchor_slots.len()];
    for step in &context.steps {
        if let Some(anchor) = step.anchor_id {
            anchor_counts[anchor_slots[&anchor]] += 1;
        }
    }
    let repeated_anchor_slots = anchor_counts.into_iter().map(|count| count > 1).collect();
    let inputs = SearchInputs {
        graph,
        destinations,
        context,
        problem,
        parameters,
        options,
        anchor_slots,
        repeated_anchor_slots,
    };
    let mut scratch = SearchScratch::new();
    if let Some(started) = build_started {
        scratch.report.build_problem_ns += started.elapsed().as_nanos() as u64;
    }
    if inputs.context.steps.len() == 2 {
        let output = search_two_step_context(&inputs, &mut scratch)?;
        if let Some(started) = started {
            scratch.report.total_search_ns += started.elapsed().as_nanos() as u64;
        }
        return Ok((output, scratch.into_report()));
    }
    let balanced_stitch_layer = (inputs.context.steps.len() - 1) as i32 / 2;
    let stitch_layer = (balanced_stitch_layer + inputs.options.stitch_bias)
        .clamp(0, inputs.context.steps.len() as i32 - 2) as usize;
    let backward = backward_beam(&inputs, &mut scratch, stitch_layer + 1)?;
    let mut backward = backward;
    extend_backward_guidance(
        &inputs,
        &mut scratch,
        stitch_layer,
        &mut backward,
        BackwardGuidanceMode::Exact,
    )?;
    if inputs.options.candidate_strategy == CandidateStrategy::SymmetricFactorMap
        && inputs.options.symmetric_message_limit > 0
    {
        extend_backward_guidance(
            &inputs,
            &mut scratch,
            stitch_layer,
            &mut backward,
            BackwardGuidanceMode::Partial,
        )?;
    }
    let (prefix_nodes, forward) = forward_beam(&inputs, &mut scratch, stitch_layer, &backward)?;
    refresh_stitch_frontier(
        &inputs,
        &mut scratch,
        stitch_layer,
        &prefix_nodes,
        &forward,
        &mut backward,
    )?;
    let stitch_started = inputs.options.profile.then(Instant::now);
    let mut completed = Vec::new();
    let home = inputs.graph.zone_index[&inputs.context.initial_zone];
    for &prefix_index in &forward {
        let prefix = &prefix_nodes[prefix_index];
        let prefix_previous = if stitch_layer == 0 {
            home
        } else {
            prefix_nodes[prefix.parent.expect("non-root forward frontier")].zone
        };
        for &suffix_index in &backward.frontiers[stitch_layer + 1] {
            let suffix = &backward.nodes[suffix_index];
            scratch.report.stitch_pairs += 1;
            if !anchors_compatible(&prefix.anchors, &suffix.anchors) {
                continue;
            }
            let boundary_score = scratch
                .local_scores
                .score(
                    inputs.scoring(),
                    stitch_layer,
                    prefix_previous,
                    prefix.zone,
                    Some(suffix.zone),
                )
                .and_then(|left| {
                    scratch
                        .local_scores
                        .score(
                            inputs.scoring(),
                            stitch_layer + 1,
                            prefix.zone,
                            suffix.zone,
                            suffix.next.map(|index| backward.nodes[index].zone),
                        )
                        .map(|right| left + right)
                });
            if let Some(boundary_score) = boundary_score {
                let score = prefix.exact_log_weight + suffix.exact_log_weight + boundary_score;
                completed.push(CompletedPlan {
                    score,
                    prefix: prefix_index,
                    suffix: suffix_index,
                });
            }
        }
    }
    if completed.is_empty() {
        scratch.report.infeasible_contexts = 1;
        return Err(SamplerError::NoFeasibleSequence {
            context_id: context.context_id,
            origin: context.initial_zone,
        });
    }
    completed.sort_unstable_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.prefix.cmp(&right.prefix))
            .then_with(|| left.suffix.cmp(&right.suffix))
    });
    if let Some(started) = stitch_started {
        scratch.report.stitch_ns += started.elapsed().as_nanos() as u64;
    }
    scratch.report.completed_plans = completed.len() as u64;
    let mut output = OutputTable::default();
    let mut seen = BTreeSet::new();
    let materialize_started = inputs.options.profile.then(Instant::now);
    for completed in completed {
        let mut zones = prefix_zones(&prefix_nodes, completed.prefix);
        zones.extend(suffix_zones(&backward.nodes, completed.suffix));
        if !seen.insert(zones.clone()) {
            continue;
        }
        append_plan(&mut output, &inputs, &zones, seen.len() as u32);
        if seen.len() == inputs.options.result_limit as usize {
            break;
        }
    }
    if let Some(started) = materialize_started {
        scratch.report.materialize_ns += started.elapsed().as_nanos() as u64;
    }
    if let Some(started) = started {
        scratch.report.total_search_ns += started.elapsed().as_nanos() as u64;
    }
    Ok((output, scratch.into_report()))
}

pub fn search_top_k_all(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    contexts: &[Context],
    parameters: Parameters,
    options: TopKOptions,
    n_threads: Option<usize>,
) -> Result<(OutputTable, TopKReport), SamplerError> {
    let compute = || {
        contexts
            .par_iter()
            .map(|context| search_context(graph, destinations, context, parameters, options))
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
    let mut report = TopKReport::default();
    for result in results {
        match result {
            Ok((table, context_report)) => {
                output.extend(table);
                report.add(&context_report);
            }
            Err(SamplerError::NoFeasibleSequence { .. }) if parameters.skip_infeasible => {
                report.contexts += 1;
                report.infeasible_contexts += 1;
            }
            Err(error) => return Err(error),
        }
    }
    Ok((output, report))
}
