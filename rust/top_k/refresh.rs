use super::*;

/// Add a small set of suffix boundary states proposed from the retained forward
/// frontier. The original backward frontier remains intact: this is a bounded
/// F-to-B seam refresh, not a replacement of reverse candidate generation.
pub(super) struct RefreshSuffixRequest<'a> {
    prefix_anchors: &'a [Option<usize>],
    candidate_slot: Option<usize>,
    candidate: usize,
    refresh_layer: usize,
    downstream: &'a [usize],
    messages: &'a BackwardMessages,
}

pub(super) fn best_refresh_suffix(
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

pub(super) fn refresh_stitch_frontier(
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
        candidate_count,
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
