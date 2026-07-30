//! Post-search neighbourhood improvement for complete plans.
//!
//! Public option and report names retain the historical `pricing_*` prefix
//! for compatibility. This is not monetary pricing: it tries exact one- and
//! two-destination replacements around the best stitched plans.

use super::*;
use std::collections::BTreeMap;

const PAIR_EXPANSION_MIN_KTH_IMPROVEMENT: f64 = 0.2;

#[derive(Clone)]
pub(super) struct RankedPlan {
    pub(super) score: f64,
    pub(super) zones: Vec<usize>,
}

struct VariableGroup {
    layers: Vec<usize>,
    activity_id: u32,
}

#[derive(Clone, Copy)]
struct GroupColumn {
    destination: usize,
    score: f64,
}

fn variable_groups(context: &Context) -> Vec<VariableGroup> {
    let mut groups = Vec::<VariableGroup>::new();
    let mut anchor_group = HashMap::<u32, usize>::new();
    for (layer, step) in context.steps.iter().enumerate() {
        if step.fixed_destination.is_some() {
            continue;
        }
        if let Some(anchor_id) = step.anchor_id {
            if let Some(&group) = anchor_group.get(&anchor_id) {
                groups[group].layers.push(layer);
            } else {
                anchor_group.insert(anchor_id, groups.len());
                groups.push(VariableGroup {
                    layers: vec![layer],
                    activity_id: step.activity_id,
                });
            }
        } else {
            groups.push(VariableGroup {
                layers: vec![layer],
                activity_id: step.activity_id,
            });
        }
    }
    groups
}

fn affected_factors(group: &VariableGroup, layer_count: usize) -> Vec<usize> {
    let mut factors = BTreeSet::new();
    for &layer in &group.layers {
        if layer > 0 {
            factors.insert(layer - 1);
        }
        factors.insert(layer);
        if layer + 1 < layer_count {
            factors.insert(layer + 1);
        }
    }
    factors.into_iter().collect()
}

fn groups_interact(left: &VariableGroup, right: &VariableGroup) -> bool {
    left.layers.iter().any(|&left_layer| {
        right
            .layers
            .iter()
            .any(|&right_layer| left_layer.abs_diff(right_layer) <= 2)
    })
}

fn pair_factors(left: &VariableGroup, right: &VariableGroup, layer_count: usize) -> Vec<usize> {
    affected_factors(left, layer_count)
        .into_iter()
        .chain(affected_factors(right, layer_count))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn apply_group(zones: &mut [usize], group: &VariableGroup, destination: usize) {
    for &layer in &group.layers {
        zones[layer] = destination;
    }
}

fn rescore_affected(inputs: &SearchInputs<'_>, zones: &[usize], factors: &[usize]) -> Option<f64> {
    let home = inputs.graph.zone_index[&inputs.context.initial_zone];
    let mut score = 0.0;
    for &layer in factors {
        let origin = if layer == 0 { home } else { zones[layer - 1] };
        score += score_local_weight(
            inputs.scoring(),
            layer,
            origin,
            zones[layer],
            zones.get(layer + 1).copied(),
        )?;
    }
    Some(score)
}

fn rank_plans(plans: &mut Vec<RankedPlan>, limit: usize) {
    plans.sort_unstable_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.zones.cmp(&right.zones))
    });
    plans.dedup_by(|left, right| left.zones == right.zones);
    plans.truncate(limit);
}

fn factor_map_columns(
    inputs: &SearchInputs<'_>,
    group: &VariableGroup,
    zones: &[usize],
    column_limit: usize,
    maps: &mut FactorMapCache,
    ranked: &mut Vec<(f64, usize)>,
) -> Result<Vec<usize>, SamplerError> {
    let layer = group.layers[0];
    let home = inputs.graph.zone_index[&inputs.context.initial_zone];
    let origin = if layer == 0 { home } else { zones[layer - 1] };
    let previous_zone = if layer == 0 {
        None
    } else if layer == 1 {
        Some(home)
    } else {
        Some(zones[layer - 2])
    };
    let next_layer = layer + 1;
    let next_next_layer = layer + 2;
    let mut suffix_nodes = Vec::with_capacity(2);
    let next_next = (next_next_layer < zones.len()).then(|| {
        suffix_nodes.push(SuffixNode {
            next: None,
            zone: zones[next_next_layer],
            exact_log_weight: 0.0,
            anchors: vec![None; inputs.anchor_layout.len()],
        });
        0
    });
    suffix_nodes.push(SuffixNode {
        next: next_next,
        zone: zones[next_layer],
        exact_log_weight: 0.0,
        anchors: vec![None; inputs.anchor_layout.len()],
    });
    let suffix_index = suffix_nodes.len() - 1;
    factor_map_candidates(
        inputs,
        FactorMapRequest {
            layer,
            previous_zone,
            origin,
            suffixes: std::slice::from_ref(&suffix_index),
            anchor_slot: None,
            anchors: &[],
            candidate_limit: column_limit.saturating_add(1),
            ranked_output: true,
        },
        &suffix_nodes,
        maps,
        ranked,
    )
}

#[allow(clippy::too_many_arguments)]
fn replacement_columns(
    inputs: &SearchInputs<'_>,
    report: &mut TopKReport,
    group: &VariableGroup,
    seed: &RankedPlan,
    base_score: f64,
    local_weights: &[f64],
    candidate_limit: usize,
    maps: &mut FactorMapCache,
    ranked: &mut Vec<(f64, usize)>,
) -> Result<Vec<GroupColumn>, SamplerError> {
    let Some(domain) = inputs.destinations.domain(group.activity_id) else {
        return Ok(Vec::new());
    };
    let original = seed.zones[group.layers[0]];
    let mut columns = Vec::with_capacity(candidate_limit);
    if group.layers.len() == 1 && group.layers[0] + 1 < seed.zones.len() {
        report.pricing_candidate_evaluations += domain.len().saturating_sub(1) as u64;
        for candidate in
            factor_map_columns(inputs, group, &seed.zones, candidate_limit, maps, ranked)?
                .into_iter()
                .filter(|&candidate| candidate != original)
                .take(candidate_limit)
        {
            let mut zones = seed.zones.clone();
            zones[group.layers[0]] = candidate;
            let Some((score, _)) = score_zones(inputs.scoring(), &zones) else {
                continue;
            };
            report.pricing_feasible_evaluations += 1;
            columns.push(GroupColumn {
                destination: candidate,
                score,
            });
        }
    } else {
        let factors = affected_factors(group, inputs.context.steps.len());
        let old_local_score = factors
            .iter()
            .map(|&factor| local_weights[factor])
            .sum::<f64>();
        let mut candidate_zones = seed.zones.clone();
        let mut neighborhood = Vec::<(f64, usize)>::with_capacity(domain.len());
        for &candidate in domain {
            if candidate == original {
                continue;
            }
            report.pricing_candidate_evaluations += 1;
            apply_group(&mut candidate_zones, group, candidate);
            let Some(new_local_score) = rescore_affected(inputs, &candidate_zones, &factors) else {
                continue;
            };
            report.pricing_feasible_evaluations += 1;
            neighborhood.push((base_score - old_local_score + new_local_score, candidate));
        }
        let compare = |left: &(f64, usize), right: &(f64, usize)| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.1.cmp(&right.1))
        };
        if neighborhood.len() > candidate_limit {
            neighborhood.select_nth_unstable_by(candidate_limit - 1, compare);
            neighborhood.truncate(candidate_limit);
        }
        neighborhood.sort_unstable_by(compare);
        for (_, candidate) in neighborhood {
            let mut zones = seed.zones.clone();
            apply_group(&mut zones, group, candidate);
            let Some((score, _)) = score_zones(inputs.scoring(), &zones) else {
                continue;
            };
            columns.push(GroupColumn {
                destination: candidate,
                score,
            });
        }
    }
    columns.sort_unstable_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.destination.cmp(&right.destination))
    });
    Ok(columns)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_pair_neighborhood(
    inputs: &SearchInputs<'_>,
    report: &mut TopKReport,
    base_score: f64,
    old_local_score: f64,
    zones: &mut [usize],
    left_group: &VariableGroup,
    right_group: &VariableGroup,
    factors: &[usize],
    left_columns: &[GroupColumn],
    right_columns: &[GroupColumn],
    limit: usize,
    probe_limit: usize,
    expansion_only: bool,
    neighborhood: &mut Vec<(f64, usize, usize)>,
    probe_scores: &mut Option<Vec<(f64, f64, bool)>>,
) {
    for (left_rank, left) in left_columns.iter().take(limit).enumerate() {
        for (right_rank, right) in right_columns.iter().take(limit).enumerate() {
            let in_probe = left_rank < probe_limit && right_rank < probe_limit;
            if expansion_only && in_probe {
                continue;
            }
            report.pricing_pair_evaluations += 1;
            apply_group(zones, left_group, left.destination);
            apply_group(zones, right_group, right.destination);
            let Some(new_local_score) = rescore_affected(inputs, zones, factors) else {
                continue;
            };
            report.pricing_pair_feasible_evaluations += 1;
            let score = base_score - old_local_score + new_local_score;
            if let Some(probe_scores) = probe_scores {
                probe_scores.push((
                    score,
                    score - (left.score + right.score - base_score),
                    in_probe,
                ));
            }
            neighborhood.push((score, left.destination, right.destination));
        }
    }
}

pub(super) fn improve_complete_plans(
    inputs: &SearchInputs<'_>,
    report: &mut TopKReport,
    factor_maps: &mut FactorMapCache,
    factor_map_ranked: &mut Vec<(f64, usize)>,
    mut current: Vec<RankedPlan>,
) -> Result<Vec<RankedPlan>, SamplerError> {
    if inputs.options.pricing_passes == 0
        || inputs.context.steps.len() < inputs.options.pricing_min_layers
        || current.is_empty()
    {
        return Ok(current);
    }
    let started = inputs.options.profile.then(Instant::now);
    let result_limit = inputs.options.result_limit as usize;
    let working_limit = inputs.options.pricing_seed_limit.max(result_limit);
    let column_limit = inputs.options.pricing_column_limit;
    let adaptive_pair_expansion = inputs.options.pricing_pair_candidate_limit > 0
        && inputs.options.pricing_pair_deep_min_layers == 0
        && inputs.options.pricing_pair_deep_candidate_limit
            > inputs.options.pricing_pair_candidate_limit;
    let pair_limit = if adaptive_pair_expansion
        || (inputs.options.pricing_pair_deep_min_layers > 0
            && inputs.context.steps.len() >= inputs.options.pricing_pair_deep_min_layers)
    {
        inputs
            .options
            .pricing_pair_candidate_limit
            .max(inputs.options.pricing_pair_deep_candidate_limit)
    } else {
        inputs.options.pricing_pair_candidate_limit
    };
    let candidate_limit = column_limit.max(pair_limit);
    let collect_pair_probes = inputs
        .options
        .active_trace
        .as_ref()
        .is_some_and(|request| request.context_id == inputs.context.context_id);
    let probe_limit = if collect_pair_probes && pair_limit > 4 {
        4
    } else {
        inputs.options.pricing_pair_candidate_limit.min(pair_limit)
    };
    rank_plans(&mut current, working_limit);
    let groups = variable_groups(inputs.context);
    let interacting_pairs = if pair_limit > 0 {
        (0..groups.len())
            .flat_map(|left| {
                ((left + 1)..groups.len()).filter_map({
                    let groups = &groups;
                    move |right| {
                        groups_interact(&groups[left], &groups[right]).then(|| {
                            (
                                left,
                                right,
                                pair_factors(
                                    &groups[left],
                                    &groups[right],
                                    inputs.context.steps.len(),
                                ),
                            )
                        })
                    }
                })
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };

    for pass in 0..inputs.options.pricing_passes {
        let previous_paths = current
            .iter()
            .map(|plan| plan.zones.clone())
            .collect::<BTreeSet<_>>();
        let seeds = current.clone();
        let mut improved = BTreeMap::<Vec<usize>, f64>::new();
        for plan in &current {
            improved.insert(plan.zones.clone(), plan.score);
        }
        let working_kth_score =
            (current.len() >= working_limit).then(|| current[working_limit - 1].score);
        for (seed_rank, seed) in seeds.iter().take(working_limit).enumerate() {
            let Some((base_score, local_weights)) = score_zones(inputs.scoring(), &seed.zones)
            else {
                continue;
            };
            let mut columns_by_group = Vec::with_capacity(groups.len());
            for group in &groups {
                let columns = replacement_columns(
                    inputs,
                    report,
                    group,
                    seed,
                    base_score,
                    &local_weights,
                    candidate_limit,
                    factor_maps,
                    factor_map_ranked,
                )?;
                for column in columns.iter().take(column_limit) {
                    let mut zones = seed.zones.clone();
                    apply_group(&mut zones, group, column.destination);
                    if improved.insert(zones, column.score).is_none() {
                        report.pricing_plans_added += 1;
                    }
                }
                columns_by_group.push(columns);
            }
            if pair_limit > 0 {
                for (left_index, right_index, factors) in &interacting_pairs {
                    let old_local_score = factors
                        .iter()
                        .map(|&factor| local_weights[factor])
                        .sum::<f64>();
                    let mut zones = seed.zones.clone();
                    let mut neighborhood = Vec::with_capacity(pair_limit * pair_limit);
                    let mut probe_scores =
                        collect_pair_probes.then(|| Vec::with_capacity(probe_limit.pow(2)));
                    let split_probe = pair_limit > probe_limit
                        && (adaptive_pair_expansion || collect_pair_probes);
                    let initial_limit = if split_probe { probe_limit } else { pair_limit };
                    evaluate_pair_neighborhood(
                        inputs,
                        report,
                        base_score,
                        old_local_score,
                        &mut zones,
                        &groups[*left_index],
                        &groups[*right_index],
                        factors,
                        &columns_by_group[*left_index],
                        &columns_by_group[*right_index],
                        initial_limit,
                        probe_limit,
                        false,
                        &mut neighborhood,
                        &mut probe_scores,
                    );
                    let probe_kth_score_improvement =
                        working_kth_score.map_or(f64::INFINITY, |kth| {
                            neighborhood
                                .iter()
                                .map(|(score, _, _)| *score)
                                .max_by(f64::total_cmp)
                                .map_or(0.0, |score| (score - kth).max(0.0))
                        });
                    let expand_pair = split_probe
                        && (!adaptive_pair_expansion
                            || probe_kth_score_improvement > PAIR_EXPANSION_MIN_KTH_IMPROVEMENT);
                    if adaptive_pair_expansion {
                        report.pricing_pair_probes += 1;
                    }
                    if expand_pair {
                        if adaptive_pair_expansion {
                            report.pricing_pair_expansions += 1;
                        }
                        evaluate_pair_neighborhood(
                            inputs,
                            report,
                            base_score,
                            old_local_score,
                            &mut zones,
                            &groups[*left_index],
                            &groups[*right_index],
                            factors,
                            &columns_by_group[*left_index],
                            &columns_by_group[*right_index],
                            pair_limit,
                            probe_limit,
                            true,
                            &mut neighborhood,
                            &mut probe_scores,
                        );
                    }
                    if let Some(probe_scores) = probe_scores {
                        let probe_evaluated = columns_by_group[*left_index].len().min(probe_limit)
                            * columns_by_group[*right_index].len().min(probe_limit);
                        let expansion_evaluated =
                            columns_by_group[*left_index].len().min(pair_limit)
                                * columns_by_group[*right_index].len().min(pair_limit)
                                - probe_evaluated;
                        let mut probe = probe_scores
                            .iter()
                            .filter(|(_, _, in_probe)| *in_probe)
                            .map(|(score, non_additivity, _)| (*score, *non_additivity))
                            .collect::<Vec<_>>();
                        probe.sort_unstable_by(|left, right| right.0.total_cmp(&left.0));
                        let boundary_score_gap = (probe.len() > working_limit)
                            .then(|| probe[working_limit - 1].0 - probe[working_limit].0);
                        let entering_working_top_k = working_kth_score.map_or(probe.len(), |kth| {
                            probe.iter().filter(|(score, _)| *score > kth).count()
                        });
                        let kth_score_improvement = working_kth_score.map_or(0.0, |kth| {
                            probe
                                .first()
                                .map_or(0.0, |(score, _)| (score - kth).max(0.0))
                        });
                        let max_non_additivity = probe
                            .iter()
                            .map(|(_, non_additivity)| *non_additivity)
                            .max_by(f64::total_cmp)
                            .unwrap_or(0.0);
                        let expansion_entering_working_top_k = working_kth_score.map_or_else(
                            || {
                                probe_scores
                                    .iter()
                                    .filter(|(_, _, in_probe)| !in_probe)
                                    .count()
                            },
                            |kth| {
                                probe_scores
                                    .iter()
                                    .filter(|(score, _, in_probe)| !in_probe && *score > kth)
                                    .count()
                            },
                        );
                        let expansion_kth_score_improvement =
                            working_kth_score.map_or(0.0, |kth| {
                                probe_scores
                                    .iter()
                                    .filter(|(_, _, in_probe)| !in_probe)
                                    .map(|(score, _, _)| *score)
                                    .max_by(f64::total_cmp)
                                    .map_or(0.0, |score| (score - kth).max(0.0))
                            });
                        report
                            .pricing_pair_probe_reports
                            .push(PricingPairProbeReport {
                                pass_index: pass,
                                seed_rank,
                                left_group: *left_index,
                                right_group: *right_index,
                                evaluated: probe_evaluated,
                                feasible: probe.len(),
                                expansion_evaluated,
                                boundary_score_gap,
                                neighborhood_saturated: probe.len() > working_limit,
                                entering_working_top_k,
                                kth_score_improvement,
                                max_non_additivity,
                                expansion_entering_working_top_k,
                                expansion_kth_score_improvement,
                            });
                    }
                    let compare = |left: &(f64, usize, usize), right: &(f64, usize, usize)| {
                        right
                            .0
                            .total_cmp(&left.0)
                            .then_with(|| left.1.cmp(&right.1))
                            .then_with(|| left.2.cmp(&right.2))
                    };
                    if neighborhood.len() > working_limit {
                        neighborhood.select_nth_unstable_by(working_limit - 1, compare);
                        neighborhood.truncate(working_limit);
                    }
                    neighborhood.sort_unstable_by(compare);
                    for (_, left, right) in neighborhood {
                        let mut zones = seed.zones.clone();
                        apply_group(&mut zones, &groups[*left_index], left);
                        apply_group(&mut zones, &groups[*right_index], right);
                        let Some((score, _)) = score_zones(inputs.scoring(), &zones) else {
                            continue;
                        };
                        if improved.insert(zones, score).is_none() {
                            report.pricing_plans_added += 1;
                            report.pricing_pair_plans_added += 1;
                        }
                    }
                }
            }
        }
        current = improved
            .into_iter()
            .map(|(zones, score)| RankedPlan { score, zones })
            .collect();
        rank_plans(&mut current, working_limit);
        report.pricing_rounds += 1;
        let new_survivors = current
            .iter()
            .filter(|plan| !previous_paths.contains(&plan.zones))
            .count();
        if pass + 1 < inputs.options.pricing_passes
            && inputs.options.pricing_next_pass_min_new > 0
            && new_survivors < inputs.options.pricing_next_pass_min_new
        {
            break;
        }
    }

    current.truncate(result_limit);
    if let Some(started) = started {
        report.pricing_ns += started.elapsed().as_nanos() as u64;
    }
    Ok(current)
}
