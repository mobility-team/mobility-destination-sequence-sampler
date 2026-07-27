use super::*;
use std::collections::BTreeMap;

#[derive(Clone)]
pub(super) struct RankedZones {
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

fn rank_plans(plans: &mut Vec<RankedZones>, limit: usize) {
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
            anchors: vec![None; inputs.anchor_slots.len()],
        });
        0
    });
    suffix_nodes.push(SuffixNode {
        next: next_next,
        zone: zones[next_layer],
        exact_log_weight: 0.0,
        anchors: vec![None; inputs.anchor_slots.len()],
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
fn price_group_columns(
    inputs: &SearchInputs<'_>,
    report: &mut TopKReport,
    group: &VariableGroup,
    seed: &RankedZones,
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

pub(super) fn price_complete_plans(
    inputs: &SearchInputs<'_>,
    report: &mut TopKReport,
    factor_maps: &mut FactorMapCache,
    factor_map_ranked: &mut Vec<(f64, usize)>,
    mut current: Vec<RankedZones>,
) -> Result<Vec<RankedZones>, SamplerError> {
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
    let pair_limit = if inputs.context.steps.len() >= inputs.options.pricing_pair_deep_min_layers {
        inputs
            .options
            .pricing_pair_candidate_limit
            .max(inputs.options.pricing_pair_deep_candidate_limit)
    } else {
        inputs.options.pricing_pair_candidate_limit
    };
    let candidate_limit = column_limit.max(pair_limit);
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
        let mut priced = BTreeMap::<Vec<usize>, f64>::new();
        for plan in &current {
            priced.insert(plan.zones.clone(), plan.score);
        }
        for seed in seeds.iter().take(working_limit) {
            let Some((base_score, local_weights)) = score_zones(inputs.scoring(), &seed.zones)
            else {
                continue;
            };
            let mut columns_by_group = Vec::with_capacity(groups.len());
            for group in &groups {
                let columns = price_group_columns(
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
                    if priced.insert(zones, column.score).is_none() {
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
                    for left in columns_by_group[*left_index].iter().take(pair_limit) {
                        for right in columns_by_group[*right_index].iter().take(pair_limit) {
                            report.pricing_pair_evaluations += 1;
                            apply_group(&mut zones, &groups[*left_index], left.destination);
                            apply_group(&mut zones, &groups[*right_index], right.destination);
                            let Some(new_local_score) = rescore_affected(inputs, &zones, factors)
                            else {
                                continue;
                            };
                            report.pricing_pair_feasible_evaluations += 1;
                            neighborhood.push((
                                base_score - old_local_score + new_local_score,
                                left.destination,
                                right.destination,
                            ));
                        }
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
                        if priced.insert(zones, score).is_none() {
                            report.pricing_plans_added += 1;
                            report.pricing_pair_plans_added += 1;
                        }
                    }
                }
            }
        }
        current = priced
            .into_iter()
            .map(|(zones, score)| RankedZones { score, zones })
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
