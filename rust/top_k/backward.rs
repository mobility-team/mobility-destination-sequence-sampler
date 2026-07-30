use super::*;

pub(super) fn backward_beam(
    inputs: &SearchInputs<'_>,
    scratch: &mut SearchScratch,
    first_layer: usize,
) -> Result<BackwardMessages, SamplerError> {
    let graph = inputs.graph;
    let destinations = inputs.destinations;
    let context = inputs.context;
    let beam_width = inputs.options.frontier_width;
    let candidate_count = inputs.options.proposal_limit_per_source;
    let anchor_layout = &inputs.anchor_layout;
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
        candidate_count,
        exploration_seed: inputs.options.exploration_seed,
    };
    let use_factor_maps = inputs.options.candidate_strategy.uses_factor_maps();
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
        anchors: vec![None; anchor_layout.len()],
    }];
    let mut frontier = vec![0];
    let terminal_layer = context.steps.len() - 1;
    if let Some(trace) = active_trace.as_mut() {
        trace.retained(terminal_layer, graph.zone_index[&terminal_zone]);
    }
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
                anchor_slot: anchor_layout.slot(layer),
                anchors: &next.anchors,
            };
            let next_next_zone = next.next.map(|index| nodes[index].zone);
            let reverse_candidates = if use_factor_maps {
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
            if let Some(trace) = active_trace.as_mut() {
                trace.proposed(layer, &reverse_candidates);
            }
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
        let selected = select_beam_indices(&scores, beam_width);
        if let Some(trace) = active_trace.as_mut() {
            for &index in &selected {
                trace.retained(layer, children[index].1);
            }
        }
        frontier = selected
            .into_iter()
            .map(|index| {
                let (next_index, destination, exact_increment) = children[index];
                let node_index = nodes.len();
                let mut anchors = nodes[next_index].anchors.clone();
                if let Some(slot) = anchor_layout.slot(layer) {
                    anchors[slot] = Some(destination);
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
        partial_anchor_candidates: vec![Vec::new(); anchor_layout.len()],
    })
}

pub(super) fn extend_backward_guidance(
    inputs: &SearchInputs<'_>,
    scratch: &mut SearchScratch,
    stitch_layer: usize,
    messages: &mut BackwardMessages,
    mode: BackwardGuidanceMode,
) -> Result<(), SamplerError> {
    let graph = inputs.graph;
    let destinations = inputs.destinations;
    let context = inputs.context;
    let continuation_state_limit = if context.steps.len() > inputs.options.factor_map_max_depth {
        inputs.options.deep_continuation_state_limit
    } else {
        inputs.options.continuation_state_limit
    };
    let guidance_width = match mode {
        BackwardGuidanceMode::Partial => inputs.options.symmetric_message_limit,
        BackwardGuidanceMode::Exact if inputs.options.candidate_strategy.uses_factor_maps() => {
            continuation_state_limit.max(4)
        }
        BackwardGuidanceMode::Exact => continuation_state_limit,
    };
    let candidate_count = inputs.options.proposal_limit_per_source;
    let anchor_layout = &inputs.anchor_layout;
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
        candidate_count,
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
    if let Some(trace) = scratch.active_trace.as_mut() {
        for &node_index in &frontier {
            trace.guidance_retained(stitch_layer + 1, messages.nodes[node_index].zone);
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
                anchor_slot: anchor_layout.slot(layer),
                anchors: &next.anchors,
            };
            let next_next_zone = next.next.map(|index| messages.nodes[index].zone);
            let use_factor_maps = mode == BackwardGuidanceMode::Partial
                || inputs.options.candidate_strategy.uses_factor_maps();
            let reverse_candidates = if use_factor_maps {
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
            if let Some(trace) = scratch.active_trace.as_mut() {
                trace.guidance_proposed(layer, &reverse_candidates);
            }
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
                        factor_map_cache.reverse_prefix_partial_calls += 1;
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
        if mode == BackwardGuidanceMode::Exact {
            if let Some(trace) = scratch.active_trace.as_mut() {
                trace.exact_guidance_rank(layer, &children, &scores);
            }
        }
        if mode == BackwardGuidanceMode::Partial {
            if let Some(slot) = anchor_layout.slot(layer) {
                if anchor_layout.repeats(slot)
                    && inputs.options.symmetric_forward_proposal_limit > 0
                {
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
        let selected = select_beam_indices(&scores, layer_width);
        frontier = selected
            .into_iter()
            .map(|index| {
                let (next_index, destination, exact_increment) = children[index];
                let node_index = messages.nodes.len();
                let mut anchors = messages.nodes[next_index].anchors.clone();
                if let Some(slot) = anchor_layout.slot(layer) {
                    anchors[slot] = Some(destination);
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
        if let Some(trace) = scratch.active_trace.as_mut() {
            for &node_index in &frontier {
                trace.guidance_retained(layer, messages.nodes[node_index].zone);
            }
        }
    }
    if let Some(started) = started {
        report.backward_guidance_ns += started.elapsed().as_nanos() as u64;
    }
    Ok(())
}
