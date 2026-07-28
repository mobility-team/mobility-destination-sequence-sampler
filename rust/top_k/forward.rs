use super::*;

pub(super) fn forward_beam(
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
    let continuation_state_limit = if context.steps.len() > inputs.options.factor_map_max_depth {
        inputs.options.deep_continuation_state_limit
    } else {
        inputs.options.continuation_state_limit
    };
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
    let active_trace = &mut scratch.active_trace;
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
                            CandidateStrategy::FactorMap
                                | CandidateStrategy::SymmetricFactorMap
                                | CandidateStrategy::AdaptiveFactorMap
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
                        ranked_output: false,
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
                            ranked_output: false,
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
            if let Some(trace) = active_trace.as_mut() {
                trace.proposed(layer, &candidate_zones);
                trace.prefix_proposed(layer, &nodes, parent_index, &candidate_zones);
            }
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
        if let Some(trace) = active_trace.as_mut() {
            for &index in &selected {
                trace.retained(layer, children[index].1);
                trace.prefix_retained(layer, &nodes, children[index].0, children[index].1);
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
