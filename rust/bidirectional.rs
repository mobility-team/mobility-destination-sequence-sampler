//! Minimal bounded bidirectional top-K search.
//!
//! This is intentionally a bounded stitch-layer beam search, not the previous all-zone
//! bidirectional DP. It uses bounded candidate lists and beam frontiers in
//! both directions, scores every locally complete rigidity-aware factor on its
//! owning front, and scores only the two factors crossing the stitch boundary when
//! stitching.

use std::collections::{BTreeSet, HashMap};
use std::time::Instant;

use rayon::prelude::*;

use crate::errors::SamplerError;
use crate::input::Context;
use crate::model::{DestinationIndex, OdGraph};
use crate::output::{OutputRow, OutputTable};
use crate::sampler::{fixed_destination_value, Parameters};
use crate::ternary_reference::{build_problem, score_local_weight, score_zones, ReferenceProblem};

mod candidates;

use candidates::{candidates, reverse_projection_candidates, CandidateCache};

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
}

#[derive(Default)]
struct LocalScoreCache {
    values: HashMap<(usize, usize, usize, Option<usize>), Option<f64>>,
}

impl LocalScoreCache {
    fn score(
        &mut self,
        graph: &OdGraph,
        destinations: &DestinationIndex,
        context: &Context,
        problem: &ReferenceProblem<'_>,
        layer: usize,
        origin: usize,
        destination: usize,
        next_destination: Option<usize>,
        parameters: Parameters,
    ) -> Option<f64> {
        let key = (layer, origin, destination, next_destination);
        if let Some(score) = self.values.get(&key) {
            return *score;
        }
        let score = score_local_weight(
            graph,
            destinations,
            context,
            problem,
            layer,
            origin,
            destination,
            next_destination,
            parameters,
        );
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
pub struct BidirectionalTopKReport {
    pub contexts: u64,
    pub forward_candidate_evaluations: u64,
    pub backward_candidate_evaluations: u64,
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
    pub seam_refresh_ns: u64,
    pub stitch_ns: u64,
    pub materialize_ns: u64,
    pub total_search_ns: u64,
}

#[derive(Clone, Copy)]
pub struct TopKOptions {
    pub frontier_width: usize,
    pub proposal_limit_per_source: usize,
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
    problem: ReferenceProblem<'a>,
    parameters: Parameters,
    options: TopKOptions,
    anchor_slots: HashMap<u32, usize>,
}

/// Per-context mutable state used by the bounded search passes.
struct SearchScratch {
    candidate_cache: CandidateCache,
    local_scores: LocalScoreCache,
    report: BidirectionalTopKReport,
}

impl SearchScratch {
    fn new() -> Self {
        Self {
            candidate_cache: CandidateCache::default(),
            local_scores: LocalScoreCache::default(),
            report: BidirectionalTopKReport {
                contexts: 1,
                ..BidirectionalTopKReport::default()
            },
        }
    }
}

impl BidirectionalTopKReport {
    fn add(&mut self, other: &Self) {
        self.contexts += other.contexts;
        self.forward_candidate_evaluations += other.forward_candidate_evaluations;
        self.backward_candidate_evaluations += other.backward_candidate_evaluations;
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

fn best_continuation_score(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    context: &Context,
    problem: &ReferenceProblem<'_>,
    layer: usize,
    previous_zone: usize,
    candidate: usize,
    prefix_anchors: &[Option<usize>],
    candidate_slot: Option<usize>,
    suffix_nodes: &[SuffixNode],
    suffix_frontier: &[usize],
    parameters: Parameters,
    local_scores: &mut LocalScoreCache,
) -> Option<f64> {
    let mut best = f64::NEG_INFINITY;
    for &suffix_index in suffix_frontier {
        let suffix = &suffix_nodes[suffix_index];
        if !candidate_anchors_compatible(prefix_anchors, &suffix.anchors, candidate_slot, candidate)
        {
            continue;
        }
        let score = local_scores
            .score(
                graph,
                destinations,
                context,
                problem,
                layer,
                previous_zone,
                candidate,
                Some(suffix.zone),
                parameters,
            )
            .and_then(|left| {
                local_scores
                    .score(
                        graph,
                        destinations,
                        context,
                        problem,
                        layer + 1,
                        candidate,
                        suffix.zone,
                        suffix.next.map(|index| suffix_nodes[index].zone),
                        parameters,
                    )
                    .map(|right| left + right + suffix.exact_log_weight)
            });
        if let Some(score) = score {
            best = best.max(score);
        }
    }
    best.is_finite().then_some(best)
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
    let problem = &inputs.problem;
    let parameters = inputs.parameters;
    let beam_width = inputs.options.frontier_width;
    let candidate_count = inputs.options.proposal_limit_per_source;
    let anchor_slots = &inputs.anchor_slots;
    let continuation_state_limit = inputs.options.continuation_state_limit;
    let continuation_proposal_limit = inputs.options.continuation_proposal_limit;
    let profile = inputs.options.profile;
    let candidate_cache = &mut scratch.candidate_cache;
    let local_scores = &mut scratch.local_scores;
    let report = &mut scratch.report;
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
        let mut pairs = Vec::new();
        for (state_index, &parent_index) in frontier.iter().enumerate() {
            let parent = &nodes[parent_index];
            let candidate_slot = context.steps[layer]
                .anchor_id
                .and_then(|anchor| anchor_slots.get(&anchor).copied());
            let mut candidate_zones = candidates(
                graph,
                destinations,
                context,
                layer,
                parent.zone,
                false,
                state_index,
                candidate_count,
                parameters.seed,
                candidate_cache,
                candidate_slot,
                &parent.anchors,
            )?;
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
            let previous_zone = if layer == 0 {
                home
            } else if layer == 1 {
                home
            } else {
                nodes[parent.parent.expect("non-root forward parent")].zone
            };
            for candidate in candidate_zones {
                report.forward_candidate_evaluations += 1;
                let local_score = if layer == 0 {
                    initial_endpoint_score(graph, destinations, context, candidate, parameters)
                } else {
                    local_scores.score(
                        graph,
                        destinations,
                        context,
                        problem,
                        layer - 1,
                        previous_zone,
                        parent.zone,
                        Some(candidate),
                        parameters,
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
                                graph,
                                destinations,
                                context,
                                problem,
                                layer,
                                parent.zone,
                                candidate,
                                &parent.anchors,
                                candidate_slot,
                                &backward.nodes,
                                &suffix_frontier
                                    [..suffix_frontier.len().min(continuation_state_limit)],
                                parameters,
                                local_scores,
                            )
                        });
                    if let Some(started) = continuation_started {
                        report.continuation_guidance_ns += started.elapsed().as_nanos() as u64;
                    }
                    children.push((
                        parent_index,
                        candidate,
                        if layer == 0 { 0.0 } else { local_score },
                    ));
                    pairs.push((parent.zone, candidate));
                    scores.push(
                        continuation_score
                            .map(|score| prefix_score + score)
                            .unwrap_or_else(|| {
                                if layer == 0 {
                                    local_score
                                } else {
                                    prefix_score
                                }
                            }),
                    );
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
            (parameters.n_draws as usize).min(beam_width),
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
fn best_refresh_suffix(
    inputs: &SearchInputs<'_>,
    local_scores: &mut LocalScoreCache,
    prefix_anchors: &[Option<usize>],
    candidate_slot: Option<usize>,
    candidate: usize,
    refresh_layer: usize,
    downstream: &[usize],
    messages: &BackwardMessages,
    unanchored_values: &mut HashMap<usize, Option<(usize, f64, f64)>>,
) -> Option<(usize, f64, f64)> {
    if inputs.anchor_slots.is_empty() {
        if let Some(value) = unanchored_values.get(&candidate) {
            return *value;
        }
    }
    let mut best = None;
    for &next_index in downstream {
        let next = &messages.nodes[next_index];
        if !inputs.anchor_slots.is_empty()
            && !candidate_anchors_compatible(
                prefix_anchors,
                &next.anchors,
                candidate_slot,
                candidate,
            )
        {
            continue;
        }
        let Some(local_score) = local_scores.score(
            inputs.graph,
            inputs.destinations,
            inputs.context,
            &inputs.problem,
            refresh_layer + 1,
            candidate,
            next.zone,
            next.next.map(|index| messages.nodes[index].zone),
            inputs.parameters,
        ) else {
            continue;
        };
        let suffix_score = next.exact_log_weight + local_score;
        if best.is_none_or(|(_, _, score)| suffix_score > score) {
            best = Some((next_index, local_score, suffix_score));
        }
    }
    if inputs.anchor_slots.is_empty() {
        unanchored_values.insert(candidate, best);
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
    let problem = &inputs.problem;
    let parameters = inputs.parameters;
    let candidate_count = inputs.options.proposal_limit_per_source;
    let refresh_per_prefix = inputs.options.seam_refresh_per_prefix;
    let anchor_slots = &inputs.anchor_slots;
    let profile = inputs.options.profile;
    let candidate_cache = &mut scratch.candidate_cache;
    let local_scores = &mut scratch.local_scores;
    let report = &mut scratch.report;
    if refresh_per_prefix == 0 {
        return Ok(());
    }
    let refresh_layer = stitch_layer + 1;
    if refresh_layer + 1 >= context.steps.len() {
        // The stitch suffix is the fixed terminal home, so there is no
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
            graph,
            destinations,
            context,
            refresh_layer,
            prefix.zone,
            false,
            state_index,
            candidate_count,
            parameters.seed,
            candidate_cache,
            candidate_slot,
            &prefix.anchors,
        )? {
            report.seam_refresh_proposals += 1;
            let best = best_refresh_suffix(
                inputs,
                local_scores,
                &prefix.anchors,
                candidate_slot,
                candidate,
                refresh_layer,
                &downstream,
                messages,
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
                graph,
                destinations,
                context,
                problem,
                stitch_layer,
                prefix_previous,
                prefix.zone,
                Some(candidate),
                parameters,
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
    let problem = &inputs.problem;
    let parameters = inputs.parameters;
    let beam_width = inputs.options.frontier_width;
    let candidate_count = inputs.options.proposal_limit_per_source;
    let anchor_slots = &inputs.anchor_slots;
    let profile = inputs.options.profile;
    let candidate_cache = &mut scratch.candidate_cache;
    let local_scores = &mut scratch.local_scores;
    let report = &mut scratch.report;
    let started = profile.then(Instant::now);
    let terminal = context.steps.last().expect("context has steps");
    let terminal_zone = terminal.fixed_destination.ok_or_else(|| {
        SamplerError::InvalidInput(format!(
            "context {} needs a fixed terminal home for bidirectional top-K search",
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
            for candidate in candidates(
                graph,
                destinations,
                context,
                layer,
                next_zone,
                true,
                state_index,
                candidate_count,
                parameters.seed,
                candidate_cache,
                context.steps[layer]
                    .anchor_id
                    .and_then(|anchor| anchor_slots.get(&anchor).copied()),
                &next.anchors,
            )? {
                report.backward_candidate_evaluations += 1;
                let score = local_scores.score(
                    graph,
                    destinations,
                    context,
                    problem,
                    layer + 1,
                    candidate,
                    next_zone,
                    next.next.map(|index| nodes[index].zone),
                    parameters,
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
            (parameters.n_draws as usize).min(beam_width),
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
    })
}

fn extend_backward_guidance(
    inputs: &SearchInputs<'_>,
    scratch: &mut SearchScratch,
    stitch_layer: usize,
    messages: &mut BackwardMessages,
) -> Result<(), SamplerError> {
    let graph = inputs.graph;
    let destinations = inputs.destinations;
    let context = inputs.context;
    let problem = &inputs.problem;
    let parameters = inputs.parameters;
    let guidance_width = inputs.options.continuation_state_limit;
    let candidate_count = inputs.options.proposal_limit_per_source;
    let anchor_slots = &inputs.anchor_slots;
    let profile = inputs.options.profile;
    let candidate_cache = &mut scratch.candidate_cache;
    let local_scores = &mut scratch.local_scores;
    let report = &mut scratch.report;
    let started = profile.then(Instant::now);
    let mut frontier = messages.frontiers[stitch_layer + 1]
        .iter()
        .copied()
        .take(guidance_width)
        .collect::<Vec<_>>();
    messages.guidance_frontiers[stitch_layer + 1] = frontier.clone();
    for layer in (0..=stitch_layer).rev() {
        let mut children = Vec::new();
        let mut scores = Vec::new();
        let mut pairs = Vec::new();
        for (state_index, &next_index) in frontier.iter().enumerate() {
            let next = &messages.nodes[next_index];
            let next_zone = next.zone;
            for candidate in candidates(
                graph,
                destinations,
                context,
                layer,
                next_zone,
                true,
                state_index,
                candidate_count,
                parameters.seed,
                candidate_cache,
                context.steps[layer]
                    .anchor_id
                    .and_then(|anchor| anchor_slots.get(&anchor).copied()),
                &next.anchors,
            )? {
                report.backward_candidate_evaluations += 1;
                let score = local_scores.score(
                    graph,
                    destinations,
                    context,
                    problem,
                    layer + 1,
                    candidate,
                    next_zone,
                    next.next.map(|index| messages.nodes[index].zone),
                    parameters,
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
            (parameters.n_draws as usize).min(guidance_width),
        );
        let children = retained
            .iter()
            .map(|&index| children[index])
            .collect::<Vec<_>>();
        let scores = retained
            .iter()
            .map(|&index| scores[index])
            .collect::<Vec<_>>();
        frontier = select_beam_indices(&scores, guidance_width)
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
        messages.guidance_frontiers[layer] = frontier.clone();
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

fn append_plan(
    output: &mut OutputTable,
    graph: &OdGraph,
    destinations: &DestinationIndex,
    context: &Context,
    problem: &crate::ternary_reference::ReferenceProblem<'_>,
    zones: &[usize],
    parameters: Parameters,
    draw_id: u32,
) {
    let Some((_, local_weights)) =
        score_zones(graph, destinations, context, problem, zones, parameters)
    else {
        return;
    };
    let mut suffixes = vec![0.0; zones.len()];
    let mut suffix = 0.0;
    for layer in (0..zones.len()).rev() {
        suffix += local_weights[layer];
        suffixes[layer] = suffix;
    }
    let mut origin = graph.zone_index[&context.initial_zone];
    for (layer, &destination) in zones.iter().enumerate() {
        output.push(OutputRow {
            context_id: context.context_id,
            draw_id,
            layer: layer as u32,
            origin: graph.zone_ids[origin],
            destination: graph.zone_ids[destination],
            local_log_weight: local_weights[layer],
            total_log_weight: suffixes[layer],
        });
        origin = destination;
    }
}

fn search_context(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    context: &Context,
    parameters: Parameters,
    options: TopKOptions,
) -> Result<(OutputTable, BidirectionalTopKReport), SamplerError> {
    let started = options.profile.then(Instant::now);
    if context.steps.len() < 3 {
        return Err(SamplerError::InvalidInput(format!(
            "context {} needs at least three steps for bidirectional top-K search",
            context.context_id
        )));
    }
    let build_started = options.profile.then(Instant::now);
    let (problem, _) = build_problem(graph, destinations, context)?;
    let inputs = SearchInputs {
        graph,
        destinations,
        context,
        problem,
        parameters,
        options,
        anchor_slots: anchor_slots(context),
    };
    let mut scratch = SearchScratch::new();
    if let Some(started) = build_started {
        scratch.report.build_problem_ns += started.elapsed().as_nanos() as u64;
    }
    let balanced_stitch_layer = (inputs.context.steps.len() - 1) as i32 / 2;
    let stitch_layer = (balanced_stitch_layer + inputs.options.stitch_bias)
        .clamp(0, inputs.context.steps.len() as i32 - 2) as usize;
    let backward = backward_beam(&inputs, &mut scratch, stitch_layer + 1)?;
    let mut backward = backward;
    extend_backward_guidance(&inputs, &mut scratch, stitch_layer, &mut backward)?;
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
                    inputs.graph,
                    inputs.destinations,
                    inputs.context,
                    &inputs.problem,
                    stitch_layer,
                    prefix_previous,
                    prefix.zone,
                    Some(suffix.zone),
                    inputs.parameters,
                )
                .and_then(|left| {
                    scratch
                        .local_scores
                        .score(
                            inputs.graph,
                            inputs.destinations,
                            inputs.context,
                            &inputs.problem,
                            stitch_layer + 1,
                            prefix.zone,
                            suffix.zone,
                            suffix.next.map(|index| backward.nodes[index].zone),
                            inputs.parameters,
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
        append_plan(
            &mut output,
            inputs.graph,
            inputs.destinations,
            inputs.context,
            &inputs.problem,
            &zones,
            inputs.parameters,
            seen.len() as u32,
        );
        if seen.len() == inputs.parameters.n_draws as usize {
            break;
        }
    }
    if let Some(started) = materialize_started {
        scratch.report.materialize_ns += started.elapsed().as_nanos() as u64;
    }
    if let Some(started) = started {
        scratch.report.total_search_ns += started.elapsed().as_nanos() as u64;
    }
    Ok((output, scratch.report))
}

pub fn search_bidirectional_top_k_all(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    contexts: &[Context],
    parameters: Parameters,
    options: TopKOptions,
    n_threads: Option<usize>,
) -> Result<(OutputTable, BidirectionalTopKReport), SamplerError> {
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
    let mut report = BidirectionalTopKReport::default();
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
