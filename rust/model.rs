use std::collections::HashMap;

use crate::errors::SamplerError;
use crate::input::{DestinationInputRow, OdCostRow};

#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub destination: usize,
    pub cost: f64,
    pub time: f64,
}

/// Shared sparse OD graph stored in compressed sparse row form.
#[derive(Debug)]
pub struct OdGraph {
    pub zone_ids: Vec<u32>,
    pub zone_index: HashMap<u32, usize>,
    pub offsets: Vec<usize>,
    pub edges: Vec<Edge>,
    edge_origins: Vec<u32>,
    outgoing_cost_edge_indices: Vec<u32>,
    incoming_offsets: Vec<usize>,
    incoming_cost_edge_indices: Vec<u32>,
    dense_edge_indices: Option<Vec<usize>>,
}

impl OdGraph {
    pub fn build(mut rows: Vec<OdCostRow>) -> Result<Self, SamplerError> {
        if rows.is_empty() {
            return Err(SamplerError::InvalidInput(
                "od_costs must contain at least one row".to_string(),
            ));
        }
        if rows
            .iter()
            .any(|row| !row.cost.is_finite() || !row.time.is_finite() || row.time < 0.0)
        {
            return Err(SamplerError::InvalidInput(
                "OD cost and time values must be finite, and time must be non-negative".to_string(),
            ));
        }

        let mut zone_ids: Vec<u32> = rows
            .iter()
            .flat_map(|row| [row.origin, row.destination])
            .collect();
        zone_ids.sort_unstable();
        zone_ids.dedup();
        let zone_index: HashMap<u32, usize> = zone_ids
            .iter()
            .enumerate()
            .map(|(index, zone_id)| (*zone_id, index))
            .collect();

        rows.sort_unstable_by_key(|row| (row.origin, row.destination));
        for pair in rows.windows(2) {
            if pair[0].origin == pair[1].origin && pair[0].destination == pair[1].destination {
                return Err(SamplerError::InvalidInput(format!(
                    "od_costs contains duplicate edge {} -> {}",
                    pair[0].origin, pair[0].destination
                )));
            }
        }

        let mut offsets = vec![0usize; zone_ids.len() + 1];
        for row in &rows {
            offsets[zone_index[&row.origin] + 1] += 1;
        }
        for index in 1..offsets.len() {
            offsets[index] += offsets[index - 1];
        }

        let edges: Vec<Edge> = rows
            .into_iter()
            .map(|row| Edge {
                destination: zone_index[&row.destination],
                cost: row.cost,
                time: row.time,
            })
            .collect();
        let mut edge_origins = vec![0u32; edges.len()];
        for origin in 0..zone_ids.len() {
            edge_origins[offsets[origin]..offsets[origin + 1]].fill(origin as u32);
        }

        let mut outgoing_cost_edge_indices: Vec<u32> = (0..edges.len())
            .map(|edge_index| edge_index as u32)
            .collect();
        for origin in 0..zone_ids.len() {
            outgoing_cost_edge_indices[offsets[origin]..offsets[origin + 1]].sort_unstable_by(
                |left, right| {
                    let left_edge = edges[*left as usize];
                    let right_edge = edges[*right as usize];
                    left_edge
                        .cost
                        .total_cmp(&right_edge.cost)
                        .then_with(|| left_edge.destination.cmp(&right_edge.destination))
                },
            );
        }

        let mut incoming_offsets = vec![0usize; zone_ids.len() + 1];
        for edge in &edges {
            incoming_offsets[edge.destination + 1] += 1;
        }
        for index in 1..incoming_offsets.len() {
            incoming_offsets[index] += incoming_offsets[index - 1];
        }
        let mut incoming_cost_edge_indices = vec![0u32; edges.len()];
        let mut next_incoming_index = incoming_offsets[..zone_ids.len()].to_vec();
        for (edge_index, edge) in edges.iter().enumerate() {
            let index = next_incoming_index[edge.destination];
            incoming_cost_edge_indices[index] = edge_index as u32;
            next_incoming_index[edge.destination] += 1;
        }
        for destination in 0..zone_ids.len() {
            incoming_cost_edge_indices
                [incoming_offsets[destination]..incoming_offsets[destination + 1]]
                .sort_unstable_by(|left, right| {
                    let left_edge = edges[*left as usize];
                    let right_edge = edges[*right as usize];
                    left_edge.cost.total_cmp(&right_edge.cost).then_with(|| {
                        edge_origins[*left as usize].cmp(&edge_origins[*right as usize])
                    })
                });
        }

        // Destination sampling repeatedly looks up a small number of exact OD
        // pairs. A dense index is cheap for near-complete matrices such as
        // Grand Genève, while sparse regional or national graphs keep the CSR
        // binary-search fallback.
        let dense_size = zone_ids.len().checked_mul(zone_ids.len());
        let dense_edge_indices = dense_size
            .filter(|&size| size <= edges.len().saturating_mul(4))
            .map(|size| {
                let mut indices = vec![usize::MAX; size];
                for origin in 0..zone_ids.len() {
                    let start = offsets[origin];
                    let end = offsets[origin + 1];
                    for (relative_index, edge) in edges[start..end].iter().enumerate() {
                        let edge_index = start + relative_index;
                        let destination = edge.destination;
                        indices[origin * zone_ids.len() + destination] = edge_index;
                    }
                }
                indices
            });

        Ok(Self {
            zone_ids,
            zone_index,
            offsets,
            edges,
            edge_origins,
            outgoing_cost_edge_indices,
            incoming_offsets,
            incoming_cost_edge_indices,
            dense_edge_indices,
        })
    }

    #[inline]
    pub fn outgoing(&self, origin: usize) -> &[Edge] {
        &self.edges[self.offsets[origin]..self.offsets[origin + 1]]
    }

    #[inline]
    pub fn outgoing_by_cost(&self, origin: usize) -> impl Iterator<Item = Edge> + '_ {
        self.outgoing_cost_edge_indices[self.offsets[origin]..self.offsets[origin + 1]]
            .iter()
            .map(|&edge_index| self.edges[edge_index as usize])
    }

    #[inline]
    pub fn incoming_by_cost(&self, destination: usize) -> impl Iterator<Item = (usize, Edge)> + '_ {
        self.incoming_cost_edge_indices
            [self.incoming_offsets[destination]..self.incoming_offsets[destination + 1]]
            .iter()
            .map(|&edge_index| {
                let edge_index = edge_index as usize;
                (
                    self.edge_origins[edge_index] as usize,
                    self.edges[edge_index],
                )
            })
    }

    #[inline]
    pub fn edge_to(&self, origin: usize, destination: usize) -> Option<Edge> {
        if let Some(indices) = &self.dense_edge_indices {
            let edge_index = indices[origin * self.zone_ids.len() + destination];
            return (edge_index != usize::MAX).then(|| self.edges[edge_index]);
        }
        self.outgoing(origin)
            .binary_search_by_key(&destination, |edge| edge.destination)
            .ok()
            .map(|index| self.outgoing(origin)[index])
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DestinationValue {
    pub log_opportunity_capacity: f64,
    pub country_value_coefficient: f64,
    pub saturation_utility: f64,
    pub shadow_price: f64,
}

#[derive(Debug)]
pub struct DestinationIndex {
    by_activity: HashMap<u32, Vec<Option<DestinationValue>>>,
    domains_by_activity: HashMap<u32, Vec<usize>>,
    attractive_by_activity: HashMap<u32, Vec<usize>>,
}

impl DestinationIndex {
    pub fn build(rows: Vec<DestinationInputRow>, graph: &OdGraph) -> Result<Self, SamplerError> {
        let mut by_activity: HashMap<u32, Vec<Option<DestinationValue>>> = HashMap::new();
        for row in rows {
            let Some(&zone) = graph.zone_index.get(&row.destination) else {
                continue;
            };
            if !row.opportunity_capacity.is_finite()
                || !row.country_value_coefficient.is_finite()
                || !row.saturation_utility.is_finite()
                || !row.shadow_price.is_finite()
                || row.opportunity_capacity < 0.0
            {
                return Err(SamplerError::InvalidInput(
                    "destination utility inputs must be finite and capacity must be non-negative"
                        .to_string(),
                ));
            }

            // One hash lookup selects the activity table. The destination scan
            // can then use direct zone-indexed reads for every OD edge.
            let activity_values = by_activity
                .entry(row.activity_id)
                .or_insert_with(|| vec![None; graph.zone_ids.len()]);
            if activity_values[zone].is_some() {
                return Err(SamplerError::InvalidInput(format!(
                    "destination_inputs contains duplicate activity_id={} destination={}",
                    row.activity_id, row.destination
                )));
            }
            activity_values[zone] = Some(DestinationValue {
                log_opportunity_capacity: if row.opportunity_capacity > 0.0 {
                    row.opportunity_capacity.ln()
                } else {
                    f64::NEG_INFINITY
                },
                country_value_coefficient: row.country_value_coefficient,
                saturation_utility: row.saturation_utility,
                shadow_price: row.shadow_price,
            });
        }
        let domains_by_activity = by_activity
            .iter()
            .map(|(&activity_id, values)| {
                let domain = values
                    .iter()
                    .enumerate()
                    .filter_map(|(zone, value)| {
                        value
                            .filter(|value| value.log_opportunity_capacity.is_finite())
                            .map(|_| zone)
                    })
                    .collect();
                (activity_id, domain)
            })
            .collect();
        // This is deliberately an iteration-level index: particle sampling
        // must not sort or scan a full activity domain for every particle.
        let attractive_by_activity = by_activity
            .iter()
            .map(|(&activity_id, values)| {
                let mut zones = values
                    .iter()
                    .enumerate()
                    .filter_map(|(zone, value)| {
                        value
                            .filter(|value| value.log_opportunity_capacity.is_finite())
                            .map(|value| {
                                // Shadow prices are the saturation signal in
                                // the particle proposal. Ranking it once is
                                // a compact approximation of the
                                // step-specific utility term.
                                (zone, value.log_opportunity_capacity + value.shadow_price)
                            })
                    })
                    .collect::<Vec<_>>();
                zones.sort_unstable_by(|(left_zone, left_value), (right_zone, right_value)| {
                    right_value
                        .total_cmp(left_value)
                        .then_with(|| left_zone.cmp(right_zone))
                });
                (
                    activity_id,
                    zones.into_iter().map(|(zone, _)| zone).collect(),
                )
            })
            .collect();
        Ok(Self {
            by_activity,
            domains_by_activity,
            attractive_by_activity,
        })
    }

    #[inline]
    pub fn activity(&self, activity_id: u32) -> Option<&[Option<DestinationValue>]> {
        self.by_activity.get(&activity_id).map(Vec::as_slice)
    }

    #[inline]
    pub fn domain(&self, activity_id: u32) -> Option<&[usize]> {
        self.domains_by_activity
            .get(&activity_id)
            .map(Vec::as_slice)
    }

    /// Capacity-ranked, saturation-ready destinations prepared once per
    /// iteration. Callers take only the bounded prefix they need.
    #[inline]
    pub fn attractive(&self, activity_id: u32) -> Option<&[usize]> {
        self.attractive_by_activity
            .get(&activity_id)
            .map(Vec::as_slice)
    }
}
