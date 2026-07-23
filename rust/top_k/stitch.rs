use super::*;

pub(super) fn prefix_zones(nodes: &[PrefixNode], mut index: usize) -> Vec<usize> {
    let mut zones = Vec::new();
    while let Some(parent) = nodes[index].parent {
        zones.push(nodes[index].zone);
        index = parent;
    }
    zones.reverse();
    zones
}

pub(super) fn suffix_zones(nodes: &[SuffixNode], mut index: usize) -> Vec<usize> {
    let mut zones = Vec::new();
    loop {
        zones.push(nodes[index].zone);
        let Some(next) = nodes[index].next else {
            return zones;
        };
        index = next;
    }
}

pub(super) struct CompletedPlan {
    score: f64,
    prefix: usize,
    suffix: usize,
}

pub(super) fn append_plan(
    output: &mut OutputTable,
    inputs: &SearchInputs<'_>,
    zones: &[usize],
    draw_id: u32,
) {
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

pub(super) fn search_two_step_context(
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

pub(super) fn search_context(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    context: &Context,
    parameters: Parameters,
    options: TopKOptions,
) -> Result<(OutputTable, TopKReport), SamplerError> {
    if options.continuation_log_gap == 0.0 {
        return search_context_once(graph, destinations, context, parameters, options);
    }
    let fallback_options = options.clone();
    match search_context_once(graph, destinations, context, parameters, options) {
        Err(SamplerError::NoFeasibleSequence { .. })
            if fallback_options.continuation_log_gap > 0.0 =>
        {
            // Score-band guidance is allowed to be selective, but it must not
            // turn a feasible bounded context into an infeasible one. Retry
            // the rare empty stitch with the fixed-width channel.
            search_context_once(
                graph,
                destinations,
                context,
                parameters,
                TopKOptions {
                    continuation_log_gap: 0.0,
                    ..fallback_options
                },
            )
        }
        result => result,
    }
}

pub(super) fn search_context_once(
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
    let active_trace = inputs
        .options
        .active_trace
        .as_ref()
        .filter(|request| request.context_id == context.context_id)
        .map(|request| ActiveTrace::new(request, graph, context))
        .transpose()?;
    let mut scratch = SearchScratch::new(inputs.options.profile, active_trace);
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
            .map(|context| {
                search_context(graph, destinations, context, parameters, options.clone())
            })
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
