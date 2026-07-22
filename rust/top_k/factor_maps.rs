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
    destination_scans: &mut u64,
    feasible_entries: &mut u64,
) -> &'a FactorScoreMap {
    match cache.entry((request.layer + 1, request.next_zone, request.next_next_zone)) {
        Entry::Occupied(entry) => {
            *hits += 1;
            entry.into_mut()
        }
        Entry::Vacant(entry) => {
            *builds += 1;
            *destination_scans += request.domain.len() as u64;
            let map = {
                let next_outbound = request
                    .next_next_zone
                    .and_then(|zone| inputs.graph.factor_edge_from(request.next_zone, zone));
                let next_departure = next_outbound.and_then(|edge| {
                    inputs
                        .context
                        .steps
                        .get(request.layer + 2)
                        .and_then(|step| {
                            adjusted_times(*step, edge).map(|(departure, _)| departure)
                        })
                });
                let mut map = FactorScoreMap::with_capacity(request.domain.len());
                for (position, &destination) in request.domain.iter().enumerate() {
                    let score = inputs
                        .graph
                        .factor_edge_to(destination, request.next_zone)
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
                        map.push(position, score);
                    }
                }
                map
            };
            *feasible_entries += map.len() as u64;
            entry.insert(map)
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
    let factor_is_exactly_scored = |factor_layer: usize| {
        known_destination(factor_layer).is_some()
            && (factor_layer == 0 || known_destination(factor_layer - 1).is_some())
            && (factor_layer + 1 == inputs.context.steps.len()
                && inputs.context.steps[factor_layer]
                    .fixed_destination
                    .is_some()
                || known_destination(factor_layer + 1).is_some())
    };
    let mut score = 0.0;
    for factor_layer in 0..=layer {
        if !factor_is_exactly_scored(factor_layer) {
            continue;
        }
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
    }
    if !factor_is_exactly_scored(0) {
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
    for known_layer in 1..=layer {
        if factor_is_exactly_scored(known_layer) || !inputs.problem.is_first_choice(known_layer) {
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
        &mut maps.next_destination_scans,
        &mut maps.next_feasible_entries,
    );
    maps.reverse_prefix_partial_calls += map.len() as u64;
    ranked.clear();
    ranked.extend((0..map.len()).filter_map(|index| {
        let (position, score) = map.entry(index);
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
                maps.previous_destination_scans += domain.len() as u64;
                let map = {
                    let inbound = inputs.graph.factor_edge_from(previous_zone, request.origin);
                    let arrival = inbound.and_then(|edge| {
                        adjusted_times(inputs.context.steps[request.layer - 1], edge)
                            .map(|(_, arrival)| arrival)
                    });
                    let mut map = FactorScoreMap::with_capacity(domain.len());
                    for (position, &destination) in domain.iter().enumerate() {
                        let score = inbound.and_then(|inbound| {
                            let outbound = inputs.graph.factor_edge_from(request.origin, destination)?;
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
                            map.push(position, score);
                        }
                    }
                    map
                };
                maps.previous_feasible_entries += map.len() as u64;
                entry.insert(map)
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
                    maps.current_destination_scans += domain.len() as u64;
                    let map = {
                        let mut map = FactorScoreMap::with_capacity(domain.len());
                        for (position, &destination) in domain.iter().enumerate() {
                            let score = inputs.graph.factor_edge_from(request.origin, destination).and_then(
                                |inbound| {
                                    inputs.graph.factor_edge_to(destination, next_zone).and_then(
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
                                map.push(position, score);
                            }
                        }
                        map
                    };
                    maps.current_feasible_entries += map.len() as u64;
                    entry.insert(map)
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
            &mut maps.next_destination_scans,
            &mut maps.next_feasible_entries,
        );
        ranked.clear();
        let mut current_index = 0;
        let mut next_index = 0;
        let mut previous_index = 0;
        while current_index < current_map.len() && next_index < next_map.len() {
            let current_position = current_map.entry(current_index).0;
            let next_position = next_map.entry(next_index).0;
            let mut position = current_position.max(next_position);
            if let Some(previous_map) = previous_map {
                if previous_index == previous_map.len() {
                    break;
                }
                position = position.max(previous_map.entry(previous_index).0);
            }
            let current = current_map.entry(current_index);
            let next = next_map.entry(next_index);
            let previous = previous_map.map(|map| map.entry(previous_index));
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
