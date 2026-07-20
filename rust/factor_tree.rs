use std::collections::{BTreeMap, HashMap};

use crate::errors::SamplerError;
use crate::input::{Context, Step};
use crate::model::{DestinationIndex, OdGraph};
use crate::output::{OutputRow, OutputTable};
use crate::profile::ContextProfile;
use crate::sampler::{fixed_destination_value, local_log_weight, Parameters};

// Status: historical exact approach retained for comparison.

#[derive(Clone, Copy, Debug)]
struct PairTransition {
    step: Step,
    origin_is_left: bool,
    suppress_capacity: bool,
}

#[derive(Debug)]
struct PairFactor {
    left: usize,
    right: usize,
    transitions: Vec<PairTransition>,
}

#[derive(Debug)]
struct TreeProblem {
    domains: Vec<Vec<usize>>,
    domain_positions: Vec<Vec<usize>>,
    step_variables: Vec<Option<usize>>,
    first_choice_by_layer: Vec<bool>,
    unary: Vec<Vec<f64>>,
    pairs: Vec<PairFactor>,
    adjacency: Vec<Vec<(usize, usize)>>,
}

#[inline]
fn add_weight(total: &mut f64, weight: Option<f64>) {
    if total.is_finite() {
        *total = weight.map_or(f64::NEG_INFINITY, |weight| *total + weight);
    }
}

fn build_problem(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    context: &Context,
    parameters: Parameters,
) -> Result<TreeProblem, SamplerError> {
    let mut anchor_variables: BTreeMap<u32, usize> = BTreeMap::new();
    let mut variable_activities: Vec<u32> = Vec::new();
    let mut step_variables: Vec<Option<usize>> = Vec::with_capacity(context.steps.len());
    let mut first_choice_by_layer = Vec::with_capacity(context.steps.len());

    for step in &context.steps {
        if step.fixed_destination.is_some() {
            step_variables.push(None);
            first_choice_by_layer.push(false);
        } else if let Some(anchor_id) = step.anchor_id {
            let variable = if let Some(&variable) = anchor_variables.get(&anchor_id) {
                if variable_activities[variable] != step.activity_id {
                    return Err(SamplerError::InvalidInput(format!(
                        "context {} uses anchor id {} for several activities",
                        context.context_id, anchor_id
                    )));
                }
                variable
            } else {
                let variable = variable_activities.len();
                variable_activities.push(step.activity_id);
                anchor_variables.insert(anchor_id, variable);
                variable
            };
            let first_choice = !step_variables.contains(&Some(variable));
            step_variables.push(Some(variable));
            first_choice_by_layer.push(first_choice);
        } else {
            let variable = variable_activities.len();
            variable_activities.push(step.activity_id);
            step_variables.push(Some(variable));
            first_choice_by_layer.push(true);
        }
    }

    let domains = variable_activities
        .iter()
        .map(|&activity_id| {
            destinations
                .domain(activity_id)
                .filter(|domain| !domain.is_empty())
                .map(<[usize]>::to_vec)
                .ok_or(SamplerError::NoFeasibleSequence {
                    context_id: context.context_id,
                    origin: context.initial_zone,
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let domain_positions = domains
        .iter()
        .map(|domain| {
            let mut positions = vec![usize::MAX; graph.zone_ids.len()];
            for (index, &zone) in domain.iter().enumerate() {
                positions[zone] = index;
            }
            positions
        })
        .collect();
    let mut unary = domains
        .iter()
        .map(|domain| vec![0.0; domain.len()])
        .collect::<Vec<_>>();
    let mut pair_by_variables: HashMap<(usize, usize), usize> = HashMap::new();
    let mut pairs: Vec<PairFactor> = Vec::new();

    let mut previous_variable: Option<usize> = None;
    let mut previous_fixed = graph.zone_index.get(&context.initial_zone).copied();
    for (layer, step) in context.steps.iter().copied().enumerate() {
        let current_variable = step_variables[layer];
        let current_fixed = step
            .fixed_destination
            .and_then(|zone_id| graph.zone_index.get(&zone_id).copied());
        if step.fixed_destination.is_some() && current_fixed.is_none() {
            return Err(SamplerError::NoFeasibleSequence {
                context_id: context.context_id,
                origin: context.initial_zone,
            });
        }
        let suppress_capacity = !first_choice_by_layer[layer];

        match (
            previous_variable,
            previous_fixed,
            current_variable,
            current_fixed,
        ) {
            (None, Some(origin), None, Some(destination)) => {
                let edge = graph.edge_to(origin, destination);
                let value = edge.and_then(|edge| {
                    local_log_weight(
                        step,
                        edge,
                        fixed_destination_value(
                            destinations.activity(step.activity_id),
                            destination,
                        ),
                        true,
                        parameters,
                    )
                });
                if value.is_none() {
                    return Err(SamplerError::NoFeasibleSequence {
                        context_id: context.context_id,
                        origin: context.initial_zone,
                    });
                }
            }
            (None, Some(origin), Some(variable), None) => {
                for (domain_index, &destination) in domains[variable].iter().enumerate() {
                    let weight = graph.edge_to(origin, destination).and_then(|edge| {
                        destinations
                            .activity(step.activity_id)
                            .and_then(|values| values[destination])
                            .and_then(|destination_value| {
                                local_log_weight(
                                    step,
                                    edge,
                                    destination_value,
                                    suppress_capacity,
                                    parameters,
                                )
                            })
                    });
                    add_weight(&mut unary[variable][domain_index], weight);
                }
            }
            (Some(variable), None, None, Some(destination)) => {
                for (domain_index, &origin) in domains[variable].iter().enumerate() {
                    let weight = graph.edge_to(origin, destination).and_then(|edge| {
                        local_log_weight(
                            step,
                            edge,
                            fixed_destination_value(
                                destinations.activity(step.activity_id),
                                destination,
                            ),
                            true,
                            parameters,
                        )
                    });
                    add_weight(&mut unary[variable][domain_index], weight);
                }
            }
            (Some(previous), None, Some(current), None) if previous == current => {
                for (domain_index, &zone) in domains[current].iter().enumerate() {
                    let weight = graph.edge_to(zone, zone).and_then(|edge| {
                        destinations
                            .activity(step.activity_id)
                            .and_then(|values| values[zone])
                            .and_then(|destination_value| {
                                local_log_weight(
                                    step,
                                    edge,
                                    destination_value,
                                    suppress_capacity,
                                    parameters,
                                )
                            })
                    });
                    add_weight(&mut unary[current][domain_index], weight);
                }
            }
            (Some(previous), None, Some(current), None) => {
                let (left, right, origin_is_left) = if previous < current {
                    (previous, current, true)
                } else {
                    (current, previous, false)
                };
                let pair_index = *pair_by_variables.entry((left, right)).or_insert_with(|| {
                    let index = pairs.len();
                    pairs.push(PairFactor {
                        left,
                        right,
                        transitions: Vec::new(),
                    });
                    index
                });
                pairs[pair_index].transitions.push(PairTransition {
                    step,
                    origin_is_left,
                    suppress_capacity,
                });
            }
            _ => {
                return Err(SamplerError::InvalidInput(format!(
                    "context {} contains an invalid destination binding",
                    context.context_id
                )));
            }
        }

        previous_variable = current_variable;
        previous_fixed = current_fixed;
    }

    let mut adjacency = vec![Vec::new(); domains.len()];
    for (pair_index, pair) in pairs.iter().enumerate() {
        adjacency[pair.left].push((pair.right, pair_index));
        adjacency[pair.right].push((pair.left, pair_index));
    }
    Ok(TreeProblem {
        domains,
        domain_positions,
        step_variables,
        first_choice_by_layer,
        unary,
        pairs,
        adjacency,
    })
}

#[allow(clippy::too_many_arguments)]
fn single_transition_message<const PROFILE: bool>(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    parameters: Parameters,
    problem: &TreeProblem,
    node: usize,
    parent: usize,
    pair: &PairFactor,
    transition: PairTransition,
    node_scores: &[f64],
    profile: &mut ContextProfile,
) -> Vec<f64> {
    let origin_variable = if transition.origin_is_left {
        pair.left
    } else {
        pair.right
    };
    let mut message = vec![f64::NEG_INFINITY; problem.domains[parent].len()];

    if origin_variable == node {
        if PROFILE {
            profile.outgoing_messages += 1;
            profile.message_edges += problem.domains[node]
                .iter()
                .enumerate()
                .filter(|(node_index, _)| node_scores[*node_index].is_finite())
                .map(|(_, &origin)| graph.outgoing_len(origin) as u64)
                .sum::<u64>();
        }
        for (node_index, &origin) in problem.domains[node].iter().enumerate() {
            let node_score = node_scores[node_index];
            if !node_score.is_finite() {
                continue;
            }
            for edge in graph.outgoing(origin) {
                let parent_index = problem.domain_positions[parent][edge.destination];
                if parent_index == usize::MAX {
                    continue;
                }
                let Some(destination_value) = destinations
                    .activity(transition.step.activity_id)
                    .and_then(|values| values[edge.destination])
                else {
                    continue;
                };
                let Some(weight) = local_log_weight(
                    transition.step,
                    *edge,
                    destination_value,
                    transition.suppress_capacity,
                    parameters,
                ) else {
                    continue;
                };
                let (left_zone, right_zone) = if pair.left == node {
                    (origin, edge.destination)
                } else {
                    (edge.destination, origin)
                };
                let Some(weight) =
                    pair.transitions[1..]
                        .iter()
                        .try_fold(weight, |total, transition| {
                            let (remaining_origin, remaining_destination) =
                                if transition.origin_is_left {
                                    (left_zone, right_zone)
                                } else {
                                    (right_zone, left_zone)
                                };
                            let edge = graph.edge_to(remaining_origin, remaining_destination)?;
                            let destination_value = destinations
                                .activity(transition.step.activity_id)?[remaining_destination]?;
                            local_log_weight(
                                transition.step,
                                edge,
                                destination_value,
                                transition.suppress_capacity,
                                parameters,
                            )
                            .map(|value| total + value)
                        })
                else {
                    continue;
                };
                message[parent_index] =
                    crate::sampler::logaddexp(message[parent_index], node_score + weight);
            }
        }
    } else {
        if PROFILE {
            profile.outgoing_messages += 1;
            profile.message_edges += problem.domains[parent]
                .iter()
                .map(|&origin| graph.outgoing_len(origin) as u64)
                .sum::<u64>();
        }
        for (parent_index, &origin) in problem.domains[parent].iter().enumerate() {
            for edge in graph.outgoing(origin) {
                let node_index = problem.domain_positions[node][edge.destination];
                if node_index == usize::MAX || !node_scores[node_index].is_finite() {
                    continue;
                }
                let Some(destination_value) = destinations
                    .activity(transition.step.activity_id)
                    .and_then(|values| values[edge.destination])
                else {
                    continue;
                };
                let Some(weight) = local_log_weight(
                    transition.step,
                    *edge,
                    destination_value,
                    transition.suppress_capacity,
                    parameters,
                ) else {
                    continue;
                };
                let (left_zone, right_zone) = if pair.left == parent {
                    (origin, edge.destination)
                } else {
                    (edge.destination, origin)
                };
                let Some(weight) =
                    pair.transitions[1..]
                        .iter()
                        .try_fold(weight, |total, transition| {
                            let (remaining_origin, remaining_destination) =
                                if transition.origin_is_left {
                                    (left_zone, right_zone)
                                } else {
                                    (right_zone, left_zone)
                                };
                            let edge = graph.edge_to(remaining_origin, remaining_destination)?;
                            let destination_value = destinations
                                .activity(transition.step.activity_id)?[remaining_destination]?;
                            local_log_weight(
                                transition.step,
                                edge,
                                destination_value,
                                transition.suppress_capacity,
                                parameters,
                            )
                            .map(|value| total + value)
                        })
                else {
                    continue;
                };
                message[parent_index] = crate::sampler::logaddexp(
                    message[parent_index],
                    node_scores[node_index] + weight,
                );
            }
        }
    }
    message
}

fn pair_weight(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    parameters: Parameters,
    pair: &PairFactor,
    left_zone: usize,
    right_zone: usize,
) -> Option<f64> {
    let mut total = 0.0;
    for transition in &pair.transitions {
        let (origin, destination) = if transition.origin_is_left {
            (left_zone, right_zone)
        } else {
            (right_zone, left_zone)
        };
        let edge = graph.edge_to(origin, destination)?;
        let destination_value = destinations.activity(transition.step.activity_id)?[destination]?;
        total += local_log_weight(
            transition.step,
            edge,
            destination_value,
            transition.suppress_capacity,
            parameters,
        )?;
    }
    Some(total)
}

fn gumbel(seed: u64, context_id: u64, draw_id: u32, variable: usize, choice: usize) -> f64 {
    let mut value = seed
        ^ context_id.wrapping_mul(0x9E3779B97F4A7C15)
        ^ u64::from(draw_id).wrapping_mul(0xBF58476D1CE4E5B9)
        ^ (variable as u64).wrapping_mul(0x94D049BB133111EB)
        ^ (choice as u64).wrapping_mul(0xD6E8FEB86659FD93);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
    value ^= value >> 31;
    let unit = ((value >> 11) as f64 + 0.5) / ((1u64 << 53) as f64);
    -(-unit.ln()).ln()
}

#[allow(clippy::too_many_arguments)]
fn upward_message<const PROFILE: bool>(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    parameters: Parameters,
    problem: &TreeProblem,
    node: usize,
    parent: usize,
    pair_index: usize,
    parents: &[Option<usize>],
    messages: &mut HashMap<(usize, usize), Vec<f64>>,
    profile: &mut ContextProfile,
) {
    for &(child, child_pair) in &problem.adjacency[node] {
        if parents[child] == Some(node) {
            upward_message::<PROFILE>(
                graph,
                destinations,
                parameters,
                problem,
                child,
                node,
                child_pair,
                parents,
                messages,
                profile,
            );
        }
    }

    let pair = &problem.pairs[pair_index];
    let node_scores = problem.unary[node]
        .iter()
        .enumerate()
        .map(|(node_index, &unary)| {
            let mut score = unary;
            for &(child, _) in &problem.adjacency[node] {
                if parents[child] == Some(node) {
                    score += messages[&(child, node)][node_index];
                }
            }
            score
        })
        .collect::<Vec<_>>();
    let message = single_transition_message::<PROFILE>(
        graph,
        destinations,
        parameters,
        problem,
        node,
        parent,
        pair,
        pair.transitions[0],
        &node_scores,
        profile,
    );
    messages.insert((node, parent), message);
}

fn choose(
    scores: impl Iterator<Item = (usize, f64)>,
    seed: u64,
    context_id: u64,
    draw_id: u32,
    variable: usize,
) -> Option<usize> {
    scores
        .filter(|(_, score)| score.is_finite())
        .map(|(index, score)| {
            (
                index,
                score + gumbel(seed, context_id, draw_id, variable, index),
            )
        })
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .map(|(index, _)| index)
}

#[allow(clippy::too_many_arguments)]
fn sample_children(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    parameters: Parameters,
    problem: &TreeProblem,
    context: &Context,
    draw_id: u32,
    node: usize,
    parents: &[Option<usize>],
    parent_pairs: &[Option<usize>],
    messages: &HashMap<(usize, usize), Vec<f64>>,
    selected: &mut [usize],
) -> Option<()> {
    for &(child, _) in &problem.adjacency[node] {
        if parents[child] != Some(node) {
            continue;
        }
        let pair = &problem.pairs[parent_pairs[child]?];
        let parent_zone = problem.domains[node][selected[node]];
        let child_index = choose(
            problem.domains[child]
                .iter()
                .enumerate()
                .map(|(child_index, &child_zone)| {
                    let mut score = problem.unary[child][child_index];
                    for &(grandchild, _) in &problem.adjacency[child] {
                        if parents[grandchild] == Some(child) {
                            score += messages[&(grandchild, child)][child_index];
                        }
                    }
                    let (left_zone, right_zone) = if pair.left == child {
                        (child_zone, parent_zone)
                    } else {
                        (parent_zone, child_zone)
                    };
                    let weight =
                        pair_weight(graph, destinations, parameters, pair, left_zone, right_zone)
                            .unwrap_or(f64::NEG_INFINITY);
                    (child_index, score + weight)
                }),
            parameters.seed,
            context.context_id,
            draw_id,
            child,
        )?;
        selected[child] = child_index;
        sample_children(
            graph,
            destinations,
            parameters,
            problem,
            context,
            draw_id,
            child,
            parents,
            parent_pairs,
            messages,
            selected,
        )?;
    }
    Some(())
}

fn sample_tree_context_impl<const PROFILE: bool>(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    context: &Context,
    parameters: Parameters,
    profile: &mut ContextProfile,
) -> Result<OutputTable, SamplerError> {
    let phase_started = PROFILE.then(std::time::Instant::now);
    let problem = build_problem(graph, destinations, context, parameters)?;
    if let Some(started) = phase_started {
        profile.tree_problem_build += started.elapsed();
        profile.variables = problem.domains.len() as u64;
        profile.pair_factors = problem.pairs.len() as u64;
        profile.pair_transitions = problem
            .pairs
            .iter()
            .map(|pair| pair.transitions.len() as u64)
            .sum();
        profile.domain_choices = problem
            .domains
            .iter()
            .map(|domain| domain.len() as u64)
            .sum();
    }

    let phase_started = PROFILE.then(std::time::Instant::now);
    let mut parents = vec![None; problem.domains.len()];
    let mut parent_pairs = vec![None; problem.domains.len()];
    let mut roots = Vec::new();
    let mut seen = vec![false; problem.domains.len()];

    for root in 0..problem.domains.len() {
        if seen[root] {
            continue;
        }
        roots.push(root);
        seen[root] = true;
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            for &(neighbor, pair_index) in &problem.adjacency[node] {
                if parents[node] == Some(neighbor) {
                    continue;
                }
                if seen[neighbor] {
                    return Err(SamplerError::CyclicContext {
                        context_id: context.context_id,
                    });
                }
                seen[neighbor] = true;
                parents[neighbor] = Some(node);
                parent_pairs[neighbor] = Some(pair_index);
                stack.push(neighbor);
            }
        }
    }
    if let Some(started) = phase_started {
        profile.tree_structure_build += started.elapsed();
    }

    let phase_started = PROFILE.then(std::time::Instant::now);
    let mut messages = HashMap::new();
    for &root in &roots {
        for &(child, pair_index) in &problem.adjacency[root] {
            if parents[child] == Some(root) {
                upward_message::<PROFILE>(
                    graph,
                    destinations,
                    parameters,
                    &problem,
                    child,
                    root,
                    pair_index,
                    &parents,
                    &mut messages,
                    profile,
                );
            }
        }
    }
    if let Some(started) = phase_started {
        profile.tree_backward += started.elapsed();
    }

    let phase_started = PROFILE.then(std::time::Instant::now);
    let initial_zone = graph.zone_index[&context.initial_zone];
    let mut output = OutputTable::default();
    for draw_id in 1..=parameters.n_draws {
        let mut selected = vec![0_usize; problem.domains.len()];
        for &root in &roots {
            selected[root] = choose(
                problem.domains[root].iter().enumerate().map(|(index, _)| {
                    let mut score = problem.unary[root][index];
                    for &(child, _) in &problem.adjacency[root] {
                        if parents[child] == Some(root) {
                            score += messages[&(child, root)][index];
                        }
                    }
                    (index, score)
                }),
                parameters.seed,
                context.context_id,
                draw_id,
                root,
            )
            .ok_or(SamplerError::NoFeasibleSequence {
                context_id: context.context_id,
                origin: context.initial_zone,
            })?;
            sample_children(
                graph,
                destinations,
                parameters,
                &problem,
                context,
                draw_id,
                root,
                &parents,
                &parent_pairs,
                &messages,
                &mut selected,
            )
            .ok_or(SamplerError::NoFeasibleSequence {
                context_id: context.context_id,
                origin: context.initial_zone,
            })?;
        }

        let mut origin = initial_zone;
        let mut destinations_by_layer = Vec::with_capacity(context.steps.len());
        let mut local_weights = Vec::with_capacity(context.steps.len());
        for (layer, step) in context.steps.iter().copied().enumerate() {
            let destination = if let Some(fixed_zone_id) = step.fixed_destination {
                graph.zone_index[&fixed_zone_id]
            } else {
                let variable = problem.step_variables[layer]
                    .expect("non-fixed steps have a destination variable");
                problem.domains[variable][selected[variable]]
            };
            let edge = graph
                .edge_to(origin, destination)
                .expect("sampled tree assignments contain feasible OD edges");
            let destination_value =
                fixed_destination_value(destinations.activity(step.activity_id), destination);
            let local = local_log_weight(
                step,
                edge,
                destination_value,
                !problem.first_choice_by_layer[layer],
                parameters,
            )
            .expect("sampled tree assignments contain feasible local utilities");
            destinations_by_layer.push(destination);
            local_weights.push(local);
            origin = destination;
        }

        let mut suffix = 0.0;
        let mut suffix_values = vec![0.0; local_weights.len()];
        for layer in (0..local_weights.len()).rev() {
            suffix += local_weights[layer];
            suffix_values[layer] = suffix;
        }
        origin = initial_zone;
        for (layer, &destination) in destinations_by_layer.iter().enumerate() {
            output.push(OutputRow {
                context_id: context.context_id,
                draw_id,
                layer: layer as u32,
                origin: graph.zone_ids[origin],
                destination: graph.zone_ids[destination],
                local_log_weight: local_weights[layer],
                total_log_weight: suffix_values[layer],
            });
            origin = destination;
        }
    }
    if let Some(started) = phase_started {
        profile.tree_forward += started.elapsed();
    }
    Ok(output)
}

pub fn sample_tree_context(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    context: &Context,
    parameters: Parameters,
) -> Result<OutputTable, SamplerError> {
    sample_tree_context_impl::<false>(
        graph,
        destinations,
        context,
        parameters,
        &mut ContextProfile::default(),
    )
}

pub fn sample_tree_context_with_profile(
    graph: &OdGraph,
    destinations: &DestinationIndex,
    context: &Context,
    parameters: Parameters,
    profile: &mut ContextProfile,
) -> Result<OutputTable, SamplerError> {
    sample_tree_context_impl::<true>(graph, destinations, context, parameters, profile)
}
