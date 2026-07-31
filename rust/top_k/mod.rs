//! Fast bounded destination-sequence search.
//!
//! The search shortlists destinations, grows partial plans from both ends,
//! joins them near the middle, and fully scores every returned plan. Child
//! modules own one phase each; shared state and invariants live here. See
//! `MODELLER_GUIDE.md` for the plain-language model.

use std::collections::{hash_map::Entry, BTreeMap, BTreeSet, HashMap};
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;

use crate::errors::SamplerError;
use crate::input::Context;
use crate::model::{DestinationIndex, OdGraph};
use crate::output::{OutputRow, OutputTable};
use crate::scoring::{
    adjusted_times, build_scoring_problem, fixed_destination_value, score_local_weight,
    score_zones, Parameters, PreparedLocalScorer, ScoringInputs, ScoringProblem,
};

mod backward;
mod candidates;
mod factor_maps;
mod forward;
mod improvement;
mod refresh;
mod stitch;

use backward::{backward_beam, extend_backward_guidance};
use candidates::{
    candidates, reverse_projection_candidates, CandidateCache, CandidateInputs, CandidateQuery,
};
use factor_maps::{
    factor_map_candidates, reverse_factor_map_candidates, reverse_prefix_partial_score,
    FactorMapRequest, ReverseFactorMapRequest,
};
use forward::forward_beam;
use improvement::{improve_complete_plans, RankedPlan};
use refresh::refresh_stitch_frontier;
pub use stitch::search_top_k_all;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CandidateStrategy {
    Heuristic,
    FactorMap,
    SymmetricFactorMap,
    AdaptiveFactorMap,
}

impl CandidateStrategy {
    pub(crate) fn parse(value: &str) -> Result<Self, SamplerError> {
        match value {
            "heuristic" => Ok(Self::Heuristic),
            "factor_map" => Ok(Self::FactorMap),
            "symmetric_factor_map" => Ok(Self::SymmetricFactorMap),
            "adaptive_factor_map" => Ok(Self::AdaptiveFactorMap),
            _ => Err(SamplerError::InvalidInput(
                "candidate_strategy must be 'factor_map', 'symmetric_factor_map', 'adaptive_factor_map', or 'heuristic'"
                    .to_string(),
            )),
        }
    }

    #[inline]
    pub(crate) fn uses_factor_maps(self) -> bool {
        matches!(
            self,
            Self::FactorMap | Self::SymmetricFactorMap | Self::AdaptiveFactorMap
        )
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

/// Fast deterministic hashing for per-context caches whose keys contain only
/// trusted compact integer indexes. These keys cannot be attacker-controlled,
/// so the collision hardening of the standard `RandomState` is unnecessary in
/// the hottest exact-score lookup path.
#[derive(Default)]
struct TrustedIndexHasher(u64);

impl TrustedIndexHasher {
    #[inline]
    fn add(&mut self, value: u64) {
        self.0 = (self.0.rotate_left(5) ^ value).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}

impl Hasher for TrustedIndexHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.add(u64::from(byte));
        }
    }

    #[inline]
    fn write_u8(&mut self, value: u8) {
        self.add(u64::from(value));
    }

    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.add(value);
    }

    #[inline]
    fn write_usize(&mut self, value: usize) {
        self.add(value as u64);
    }
}

type TrustedIndexMap<K, V> = HashMap<K, V, BuildHasherDefault<TrustedIndexHasher>>;
type LocalScoreMap = TrustedIndexMap<(usize, usize, usize, Option<usize>), Option<f64>>;

#[derive(Clone, Copy, Eq, PartialEq)]
enum BackwardGuidanceMode {
    Exact,
    Partial,
}

#[derive(Default)]
struct LocalScoreCache {
    values: LocalScoreMap,
    profile: bool,
    hits: u64,
    builds: u64,
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
    previous_destination_scans: u64,
    current_destination_scans: u64,
    next_destination_scans: u64,
    previous_feasible_entries: u64,
    current_feasible_entries: u64,
    next_feasible_entries: u64,
    reverse_prefix_partial_calls: u64,
}

/// Feasible factor scores, aligned to an activity-domain position rather than
/// the global zone index. Infeasible destinations are omitted completely.
enum FactorPositions {
    U16(Vec<u16>),
    Usize(Vec<usize>),
}

struct FactorScoreMap {
    positions: FactorPositions,
    scores: Vec<f64>,
}

impl FactorScoreMap {
    fn with_capacity(domain_len: usize) -> Self {
        let positions = if u16::try_from(domain_len.saturating_sub(1)).is_ok() {
            FactorPositions::U16(Vec::with_capacity(domain_len))
        } else {
            FactorPositions::Usize(Vec::with_capacity(domain_len))
        };
        Self {
            positions,
            scores: Vec::with_capacity(domain_len),
        }
    }

    fn push(&mut self, position: usize, score: f64) {
        match &mut self.positions {
            FactorPositions::U16(positions) => positions.push(position as u16),
            FactorPositions::Usize(positions) => positions.push(position),
        }
        self.scores.push(score);
    }

    fn len(&self) -> usize {
        self.scores.len()
    }

    fn entry(&self, index: usize) -> (usize, f64) {
        let position = match &self.positions {
            FactorPositions::U16(positions) => positions[index] as usize,
            FactorPositions::Usize(positions) => positions[index],
        };
        (position, self.scores[index])
    }
}

impl LocalScoreCache {
    fn new(profile: bool) -> Self {
        Self {
            profile,
            ..Self::default()
        }
    }

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
            if self.profile {
                self.hits += 1;
            }
            return *score;
        }
        if self.profile {
            self.builds += 1;
        }
        let score = score_local_weight(inputs, layer, origin, destination, next_destination);
        self.values.insert(key, score);
        score
    }
}

/// Compact anchor bookkeeping prepared once for a context.
///
/// Search passes operate on sequence positions, so they should not repeatedly
/// hash the transport-model `anchor_id`. Each step instead gets an optional
/// compact slot into the anchor assignment vectors carried by search states.
struct AnchorLayout {
    step_slots: Vec<Option<usize>>,
    repeated_slots: Vec<bool>,
}

impl AnchorLayout {
    fn build(context: &Context) -> Self {
        let mut slots_by_id = HashMap::new();
        let mut step_slots = Vec::with_capacity(context.steps.len());
        let mut counts = Vec::<u32>::new();
        for step in &context.steps {
            let slot = step.anchor_id.map(|anchor_id| {
                let next_slot = slots_by_id.len();
                let slot = *slots_by_id.entry(anchor_id).or_insert(next_slot);
                if slot == counts.len() {
                    counts.push(0);
                }
                counts[slot] += 1;
                slot
            });
            step_slots.push(slot);
        }
        Self {
            step_slots,
            repeated_slots: counts.into_iter().map(|count| count > 1).collect(),
        }
    }

    fn len(&self) -> usize {
        self.repeated_slots.len()
    }

    fn is_empty(&self) -> bool {
        self.repeated_slots.is_empty()
    }

    fn slot(&self, layer: usize) -> Option<usize> {
        self.step_slots[layer]
    }

    fn repeats(&self, slot: usize) -> bool {
        self.repeated_slots[slot]
    }

    fn has_repeated(&self) -> bool {
        self.repeated_slots.iter().any(|&repeated| repeated)
    }
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
    pub factor_map_destination_evaluations: u64,
    pub factor_map_previous_hits: u64,
    pub factor_map_previous_builds: u64,
    pub factor_map_current_hits: u64,
    pub factor_map_current_builds: u64,
    pub factor_map_next_hits: u64,
    pub factor_map_next_builds: u64,
    pub factor_map_previous_destination_scans: u64,
    pub factor_map_current_destination_scans: u64,
    pub factor_map_next_destination_scans: u64,
    pub factor_map_previous_feasible_entries: u64,
    pub factor_map_current_feasible_entries: u64,
    pub factor_map_next_feasible_entries: u64,
    pub reverse_prefix_partial_calls: u64,
    pub local_score_cache_hits: u64,
    pub local_score_cache_builds: u64,
    pub continuation_proposals: u64,
    pub seam_refresh_proposals: u64,
    pub seam_refresh_states: u64,
    pub pricing_candidate_evaluations: u64,
    pub pricing_feasible_evaluations: u64,
    pub pricing_plans_added: u64,
    pub pricing_pair_evaluations: u64,
    pub pricing_pair_feasible_evaluations: u64,
    pub pricing_pair_plans_added: u64,
    pub pricing_pair_probes: u64,
    pub pricing_pair_expansions: u64,
    pub pricing_pair_probe_reports: Vec<PricingPairProbeReport>,
    pub pricing_rounds: u64,
    pub stitch_pairs: u64,
    pub completed_plans: u64,
    /// Contexts for which bounded search returned no complete feasible plan.
    /// This is not proof that the model inputs admit no feasible plan.
    pub contexts_without_plan: u64,
    pub build_problem_ns: u64,
    pub backward_search_ns: u64,
    pub backward_guidance_ns: u64,
    pub forward_search_ns: u64,
    pub continuation_guidance_ns: u64,
    pub factor_map_ns: u64,
    pub seam_refresh_ns: u64,
    pub pricing_ns: u64,
    pub stitch_ns: u64,
    pub materialize_ns: u64,
    pub total_search_ns: u64,
    pub active_trace_targets: Vec<ActiveTraceTargetReport>,
}

#[derive(Debug, Clone)]
pub struct PricingPairProbeReport {
    pub pass_index: usize,
    pub seed_rank: usize,
    pub left_group: usize,
    pub right_group: usize,
    pub evaluated: usize,
    pub feasible: usize,
    pub expansion_evaluated: usize,
    pub boundary_score_gap: Option<f64>,
    pub neighborhood_saturated: bool,
    pub entering_working_top_k: usize,
    pub kth_score_improvement: f64,
    pub max_non_additivity: f64,
    pub expansion_entering_working_top_k: usize,
    pub expansion_kth_score_improvement: f64,
}

#[derive(Clone, Debug)]
pub struct ActiveTraceRequest {
    pub context_id: u64,
    pub target_plans: Arc<[Vec<u32>]>,
}

#[derive(Debug, Clone)]
pub struct ActiveTraceTargetReport {
    pub zones: Vec<u32>,
    pub proposed: Vec<bool>,
    pub retained: Vec<bool>,
    /// Whether the exact target prefix through each layer entered that
    /// layer's candidate set. Unlike `proposed`, this keeps parent lineage.
    pub prefix_proposed: Vec<Option<bool>>,
    /// Whether the exact target prefix through each layer survived the
    /// forward frontier. Unlike `retained`, this keeps parent lineage.
    pub prefix_retained: Vec<Option<bool>>,
    /// Whether a target zone appeared in either retained reverse-guidance
    /// frontier at the corresponding layer.
    pub guidance_retained: Vec<bool>,
    /// Whether a target zone entered either reverse-guidance candidate pool
    /// at the corresponding layer.
    pub guidance_proposed: Vec<bool>,
    /// Best rank of a target zone in the exact reverse-guidance candidate
    /// score at each layer; `None` means it was never proposed there.
    pub exact_guidance_rank: Vec<Option<usize>>,
    /// Score loss of the best target-zone reverse-guidance child relative to
    /// the best child at that layer. This exposes whether a missed target was
    /// narrowly or decisively below the one-state continuation channel.
    pub exact_guidance_log_gap: Vec<Option<f64>>,
}

#[derive(Clone)]
pub struct TopKOptions {
    pub exploration_seed: u64,
    pub result_limit: u32,
    pub frontier_width: usize,
    pub proposal_limit_per_source: usize,
    pub symmetric_message_limit: usize,
    pub symmetric_state_limit: usize,
    pub symmetric_forward_proposal_limit: usize,
    pub candidate_strategy: CandidateStrategy,
    pub factor_map_max_depth: usize,
    pub stitch_bias: i32,
    pub continuation_state_limit: usize,
    pub deep_continuation_state_limit: usize,
    pub continuation_proposal_limit: usize,
    pub seam_refresh_per_prefix: usize,
    /// Exact single-choice improvement rounds over complete plans.
    /// Zero disables this phase.
    pub pricing_passes: usize,
    /// Complete plans retained between improvement rounds.
    pub pricing_seed_limit: usize,
    /// Best new plans retained from each single-choice neighborhood.
    pub pricing_column_limit: usize,
    /// Best exact conditional replacements crossed for each interacting pair.
    /// Zero disables the experimental two-variable neighborhood.
    pub pricing_pair_candidate_limit: usize,
    /// Wider pair candidate budget for local or depth-routed expansion.
    pub pricing_pair_deep_candidate_limit: usize,
    /// Zero enables local probe-and-expand; positive values retain a depth comparator.
    pub pricing_pair_deep_min_layers: usize,
    /// Require this many newly improved plans to survive before another round.
    /// Zero runs every requested improvement round.
    pub pricing_next_pass_min_new: usize,
    /// Do not improve contexts shallower than this many destination layers.
    pub pricing_min_layers: usize,
    pub profile: bool,
    pub active_trace: Option<ActiveTraceRequest>,
}

impl TopKOptions {
    pub(crate) fn active_incumbent(result_limit: u32) -> Self {
        Self {
            exploration_seed: 42,
            result_limit,
            frontier_width: 40,
            proposal_limit_per_source: 16,
            symmetric_message_limit: 4,
            symmetric_state_limit: 4,
            symmetric_forward_proposal_limit: 20,
            candidate_strategy: CandidateStrategy::AdaptiveFactorMap,
            factor_map_max_depth: 5,
            stitch_bias: 1,
            continuation_state_limit: 1,
            deep_continuation_state_limit: 2,
            continuation_proposal_limit: 1,
            seam_refresh_per_prefix: 1,
            pricing_passes: 2,
            pricing_seed_limit: 10,
            pricing_column_limit: 4,
            pricing_pair_candidate_limit: 4,
            pricing_pair_deep_candidate_limit: 8,
            pricing_pair_deep_min_layers: 0,
            pricing_next_pass_min_new: 3,
            pricing_min_layers: 6,
            profile: false,
            active_trace: None,
        }
    }
}

struct ActiveTrace {
    targets: Vec<(Vec<usize>, ActiveTraceTargetReport)>,
}

impl ActiveTrace {
    fn new(
        request: &ActiveTraceRequest,
        graph: &OdGraph,
        context: &Context,
    ) -> Result<Self, SamplerError> {
        let mut targets = Vec::with_capacity(request.target_plans.len());
        for zones in request.target_plans.iter() {
            if zones.len() != context.steps.len() {
                return Err(SamplerError::InvalidInput(format!(
                    "active trace target for context {} has {} zones; expected {}",
                    context.context_id,
                    zones.len(),
                    context.steps.len()
                )));
            }
            let internal = zones
                .iter()
                .map(|zone| {
                    graph.zone_index.get(zone).copied().ok_or_else(|| {
                        SamplerError::InvalidInput(format!(
                            "active trace target references unknown zone {zone}"
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            targets.push((
                internal,
                ActiveTraceTargetReport {
                    zones: zones.clone(),
                    proposed: vec![false; zones.len()],
                    retained: vec![false; zones.len()],
                    prefix_proposed: vec![None; zones.len()],
                    prefix_retained: vec![None; zones.len()],
                    guidance_retained: vec![false; zones.len()],
                    guidance_proposed: vec![false; zones.len()],
                    exact_guidance_rank: vec![None; zones.len()],
                    exact_guidance_log_gap: vec![None; zones.len()],
                },
            ));
        }
        Ok(Self { targets })
    }

    fn proposed(&mut self, layer: usize, candidates: &[usize]) {
        for (targets, report) in &mut self.targets {
            report.proposed[layer] |= candidates.contains(&targets[layer]);
        }
    }

    fn retained(&mut self, layer: usize, destination: usize) {
        for (targets, report) in &mut self.targets {
            report.retained[layer] |= targets[layer] == destination;
        }
    }

    fn prefix_matches(
        nodes: &[PrefixNode],
        mut node_index: usize,
        target: &[usize],
        prefix_len: usize,
    ) -> bool {
        for expected_layer in (0..prefix_len).rev() {
            let node = &nodes[node_index];
            if node.parent.is_none() || node.zone != target[expected_layer] {
                return false;
            }
            node_index = node.parent.expect("non-root prefix node has a parent");
        }
        true
    }

    fn prefix_proposed(
        &mut self,
        layer: usize,
        nodes: &[PrefixNode],
        parent_index: usize,
        candidates: &[usize],
    ) {
        for (target, report) in &mut self.targets {
            let matches_target = candidates.contains(&target[layer])
                && Self::prefix_matches(nodes, parent_index, target, layer);
            report.prefix_proposed[layer] =
                Some(report.prefix_proposed[layer].unwrap_or(false) || matches_target);
        }
    }

    fn prefix_retained(
        &mut self,
        layer: usize,
        nodes: &[PrefixNode],
        parent_index: usize,
        destination: usize,
    ) {
        for (target, report) in &mut self.targets {
            let matches_target = destination == target[layer]
                && Self::prefix_matches(nodes, parent_index, target, layer);
            report.prefix_retained[layer] =
                Some(report.prefix_retained[layer].unwrap_or(false) || matches_target);
        }
    }

    fn guidance_retained(&mut self, layer: usize, destination: usize) {
        for (targets, report) in &mut self.targets {
            report.guidance_retained[layer] |= targets[layer] == destination;
        }
    }

    fn guidance_proposed(&mut self, layer: usize, candidates: &[usize]) {
        for (targets, report) in &mut self.targets {
            report.guidance_proposed[layer] |= candidates.contains(&targets[layer]);
        }
    }

    fn exact_guidance_rank(
        &mut self,
        layer: usize,
        children: &[(usize, usize, f64)],
        scores: &[f64],
    ) {
        let mut ranked = scores.iter().copied().enumerate().collect::<Vec<_>>();
        ranked.sort_unstable_by(|(left_index, left), (right_index, right)| {
            right
                .total_cmp(left)
                .then_with(|| left_index.cmp(right_index))
        });
        for (target, report) in &mut self.targets {
            let target = ranked
                .iter()
                .find(|(index, _)| children[*index].1 == target[layer])
                .copied();
            if let Some((target_index, target_score)) = target {
                let rank = ranked
                    .iter()
                    .position(|(index, _)| *index == target_index)
                    .expect("target score came from ranked children")
                    + 1;
                report.exact_guidance_rank[layer] = Some(
                    report.exact_guidance_rank[layer].map_or(rank, |existing| existing.min(rank)),
                );
                let gap = ranked[0].1 - target_score;
                report.exact_guidance_log_gap[layer] = Some(
                    report.exact_guidance_log_gap[layer].map_or(gap, |existing| existing.min(gap)),
                );
            }
        }
    }

    fn into_reports(self) -> Vec<ActiveTraceTargetReport> {
        self.targets.into_iter().map(|(_, report)| report).collect()
    }
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
    prepared_factor_scorers: Vec<PreparedLocalScorer<'a>>,
    anchor_layout: AnchorLayout,
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
    active_trace: Option<ActiveTrace>,
}

impl SearchScratch {
    fn new(profile: bool, active_trace: Option<ActiveTrace>) -> Self {
        Self {
            candidate_cache: CandidateCache::default(),
            factor_map_cache: FactorMapCache::default(),
            factor_map_ranked: Vec::new(),
            local_scores: LocalScoreCache::new(profile),
            report: TopKReport {
                contexts: 1,
                ..TopKReport::default()
            },
            active_trace,
        }
    }

    fn into_report(mut self) -> TopKReport {
        if let Some(trace) = self.active_trace.take() {
            self.report.active_trace_targets = trace.into_reports();
        }
        self.report.factor_map_previous_hits = self.factor_map_cache.previous_hits;
        self.report.factor_map_previous_builds = self.factor_map_cache.previous_builds;
        self.report.factor_map_current_hits = self.factor_map_cache.current_hits;
        self.report.factor_map_current_builds = self.factor_map_cache.current_builds;
        self.report.factor_map_next_hits = self.factor_map_cache.next_hits;
        self.report.factor_map_next_builds = self.factor_map_cache.next_builds;
        self.report.factor_map_previous_destination_scans =
            self.factor_map_cache.previous_destination_scans;
        self.report.factor_map_current_destination_scans =
            self.factor_map_cache.current_destination_scans;
        self.report.factor_map_next_destination_scans =
            self.factor_map_cache.next_destination_scans;
        self.report.factor_map_previous_feasible_entries =
            self.factor_map_cache.previous_feasible_entries;
        self.report.factor_map_current_feasible_entries =
            self.factor_map_cache.current_feasible_entries;
        self.report.factor_map_next_feasible_entries = self.factor_map_cache.next_feasible_entries;
        self.report.reverse_prefix_partial_calls =
            self.factor_map_cache.reverse_prefix_partial_calls;
        self.report.local_score_cache_hits = self.local_scores.hits;
        self.report.local_score_cache_builds = self.local_scores.builds;
        self.report
    }
}

impl TopKReport {
    fn add(&mut self, other: &Self) {
        self.contexts += other.contexts;
        self.forward_candidate_evaluations += other.forward_candidate_evaluations;
        self.backward_candidate_evaluations += other.backward_candidate_evaluations;
        self.factor_map_destination_evaluations += other.factor_map_destination_evaluations;
        self.factor_map_previous_hits += other.factor_map_previous_hits;
        self.factor_map_previous_builds += other.factor_map_previous_builds;
        self.factor_map_current_hits += other.factor_map_current_hits;
        self.factor_map_current_builds += other.factor_map_current_builds;
        self.factor_map_next_hits += other.factor_map_next_hits;
        self.factor_map_next_builds += other.factor_map_next_builds;
        self.factor_map_previous_destination_scans += other.factor_map_previous_destination_scans;
        self.factor_map_current_destination_scans += other.factor_map_current_destination_scans;
        self.factor_map_next_destination_scans += other.factor_map_next_destination_scans;
        self.factor_map_previous_feasible_entries += other.factor_map_previous_feasible_entries;
        self.factor_map_current_feasible_entries += other.factor_map_current_feasible_entries;
        self.factor_map_next_feasible_entries += other.factor_map_next_feasible_entries;
        self.reverse_prefix_partial_calls += other.reverse_prefix_partial_calls;
        self.local_score_cache_hits += other.local_score_cache_hits;
        self.local_score_cache_builds += other.local_score_cache_builds;
        self.continuation_proposals += other.continuation_proposals;
        self.seam_refresh_proposals += other.seam_refresh_proposals;
        self.seam_refresh_states += other.seam_refresh_states;
        self.pricing_candidate_evaluations += other.pricing_candidate_evaluations;
        self.pricing_feasible_evaluations += other.pricing_feasible_evaluations;
        self.pricing_plans_added += other.pricing_plans_added;
        self.pricing_pair_evaluations += other.pricing_pair_evaluations;
        self.pricing_pair_feasible_evaluations += other.pricing_pair_feasible_evaluations;
        self.pricing_pair_plans_added += other.pricing_pair_plans_added;
        self.pricing_pair_probes += other.pricing_pair_probes;
        self.pricing_pair_expansions += other.pricing_pair_expansions;
        self.pricing_pair_probe_reports
            .extend(other.pricing_pair_probe_reports.iter().cloned());
        self.pricing_rounds += other.pricing_rounds;
        self.stitch_pairs += other.stitch_pairs;
        self.completed_plans += other.completed_plans;
        self.contexts_without_plan += other.contexts_without_plan;
        self.build_problem_ns += other.build_problem_ns;
        self.backward_search_ns += other.backward_search_ns;
        self.backward_guidance_ns += other.backward_guidance_ns;
        self.forward_search_ns += other.forward_search_ns;
        self.continuation_guidance_ns += other.continuation_guidance_ns;
        self.factor_map_ns += other.factor_map_ns;
        self.seam_refresh_ns += other.seam_refresh_ns;
        self.pricing_ns += other.pricing_ns;
        self.stitch_ns += other.stitch_ns;
        self.materialize_ns += other.materialize_ns;
        self.total_search_ns += other.total_search_ns;
        self.active_trace_targets
            .extend(other.active_trace_targets.iter().cloned());
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
