//! Bounded bidirectional top-K search.
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

// Search passes are physically separated to keep task-scoped reads small.
include!("factor_maps.rs");
include!("forward.rs");
include!("refresh.rs");
include!("backward.rs");
include!("stitch.rs");
