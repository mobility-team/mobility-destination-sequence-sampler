//! Runtime check for the hierarchical travel-kernel experiment.
//!
//! Factor construction stays in the Python research script. This module only
//! measures the arithmetic that a production Rust implementation would own.
//!
//! Status: paused research experiment; not the active redesign.

use std::time::{Duration, Instant};

use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

enum MatrixBlock {
    Exact {
        left: Vec<usize>,
        right: Vec<usize>,
        values: Vec<f64>,
    },
    LowRank {
        left: Vec<usize>,
        right: Vec<usize>,
        rank: usize,
        left_factor: Vec<f64>,
        right_factor: Vec<f64>,
    },
}

impl MatrixBlock {
    fn parse(
        (is_low_rank, left, right, rank, first_values, second_values): (
            bool,
            Vec<usize>,
            Vec<usize>,
            usize,
            Vec<f64>,
            Vec<f64>,
        ),
    ) -> PyResult<Self> {
        if left.is_empty() || right.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "hierarchical blocks cannot have empty index sets",
            ));
        }
        if is_low_rank {
            if rank == 0
                || first_values.len() != left.len() * rank
                || second_values.len() != rank * right.len()
            {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "low-rank block dimensions do not match its factors",
                ));
            }
            Ok(Self::LowRank {
                left,
                right,
                rank,
                left_factor: first_values,
                right_factor: second_values,
            })
        } else {
            if first_values.len() != left.len() * right.len() || !second_values.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "exact block dimensions do not match its values",
                ));
            }
            Ok(Self::Exact {
                left,
                right,
                values: first_values,
            })
        }
    }

    fn rank(&self) -> usize {
        match self {
            Self::Exact { .. } => 0,
            Self::LowRank { rank, .. } => *rank,
        }
    }

    fn add_product(&self, input: &[f64], output: &mut [f64], intermediate: &mut [f64]) {
        match self {
            Self::Exact {
                left,
                right,
                values,
            } => {
                for (left_offset, &origin) in left.iter().enumerate() {
                    let row = &values[left_offset * right.len()..(left_offset + 1) * right.len()];
                    let value = row
                        .iter()
                        .zip(right)
                        .map(|(&coefficient, &destination)| coefficient * input[destination])
                        .sum::<f64>();
                    output[origin] += value;
                }
            }
            Self::LowRank {
                left,
                right,
                rank,
                left_factor,
                right_factor,
            } => {
                for component in 0..*rank {
                    let row = &right_factor[component * right.len()..(component + 1) * right.len()];
                    intermediate[component] = row
                        .iter()
                        .zip(right)
                        .map(|(&coefficient, &destination)| coefficient * input[destination])
                        .sum();
                }
                for (left_offset, &origin) in left.iter().enumerate() {
                    let row = &left_factor[left_offset * *rank..(left_offset + 1) * *rank];
                    output[origin] += row
                        .iter()
                        .zip(intermediate.iter())
                        .map(|(&coefficient, &value)| coefficient * value)
                        .sum::<f64>();
                }
            }
        }
    }
}

fn dense_product(kernel: &[f64], n_zones: usize, input: &[f64]) -> Vec<f64> {
    kernel
        .chunks_exact(n_zones)
        .map(|row| {
            row.iter()
                .zip(input)
                .map(|(&coefficient, &value)| coefficient * value)
                .sum()
        })
        .collect()
}

fn hierarchical_product(
    blocks: &[MatrixBlock],
    maximum_rank: usize,
    n_zones: usize,
    input: &[f64],
) -> Vec<f64> {
    let mut output = vec![0.0; n_zones];
    let mut intermediate = vec![0.0; maximum_rank];
    for block in blocks {
        block.add_product(input, &mut output, &mut intermediate);
    }
    output
}

fn median_duration(mut durations: Vec<Duration>) -> Duration {
    durations.sort_unstable();
    durations[durations.len() / 2]
}

#[pyfunction]
#[pyo3(signature = (*, n_zones, dense_kernel, blocks, n_right_hand_sides, repetitions=5))]
#[allow(clippy::type_complexity)]
pub(crate) fn benchmark_hierarchical_kernel(
    py: Python<'_>,
    n_zones: usize,
    dense_kernel: Vec<f64>,
    blocks: Vec<(bool, Vec<usize>, Vec<usize>, usize, Vec<f64>, Vec<f64>)>,
    n_right_hand_sides: usize,
    repetitions: usize,
) -> PyResult<PyObject> {
    if n_zones == 0 || n_right_hand_sides == 0 || repetitions == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "zone, right-hand-side, and repetition counts must be positive",
        ));
    }
    if dense_kernel.len() != n_zones * n_zones {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "dense kernel dimensions do not match n_zones",
        ));
    }
    let blocks = blocks
        .into_iter()
        .map(MatrixBlock::parse)
        .collect::<PyResult<Vec<_>>>()?;
    let maximum_rank = blocks.iter().map(MatrixBlock::rank).max().unwrap_or(0);
    let inputs = (0..n_right_hand_sides)
        .map(|right_hand_side| {
            (0..n_zones)
                .map(|zone| {
                    let value =
                        (zone.wrapping_mul(636_413) + right_hand_side.wrapping_mul(1_442_695) + 17)
                            % 10_007;
                    (value + 1) as f64 / 10_008.0
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let (dense_seconds, hierarchical_seconds, maximum_absolute_error, checksum_difference) = py
        .allow_threads(|| {
            let dense_reference = inputs
                .par_iter()
                .map(|input| dense_product(&dense_kernel, n_zones, input))
                .collect::<Vec<_>>();
            let hierarchical_reference = inputs
                .par_iter()
                .map(|input| hierarchical_product(&blocks, maximum_rank, n_zones, input))
                .collect::<Vec<_>>();
            let maximum_absolute_error = dense_reference
                .iter()
                .flatten()
                .zip(hierarchical_reference.iter().flatten())
                .map(|(&exact, &approximate)| (exact - approximate).abs())
                .fold(0.0, f64::max);
            let dense_checksum = dense_reference.iter().flatten().sum::<f64>();
            let hierarchical_checksum = hierarchical_reference.iter().flatten().sum::<f64>();

            let mut dense_durations = Vec::with_capacity(repetitions);
            let mut hierarchical_durations = Vec::with_capacity(repetitions);
            for _ in 0..repetitions {
                let started = Instant::now();
                let output = inputs
                    .par_iter()
                    .map(|input| dense_product(&dense_kernel, n_zones, input))
                    .collect::<Vec<_>>();
                std::hint::black_box(output);
                dense_durations.push(started.elapsed());

                let started = Instant::now();
                let output = inputs
                    .par_iter()
                    .map(|input| hierarchical_product(&blocks, maximum_rank, n_zones, input))
                    .collect::<Vec<_>>();
                std::hint::black_box(output);
                hierarchical_durations.push(started.elapsed());
            }
            (
                median_duration(dense_durations).as_secs_f64(),
                median_duration(hierarchical_durations).as_secs_f64(),
                maximum_absolute_error,
                (dense_checksum - hierarchical_checksum).abs(),
            )
        });

    let report = PyDict::new(py);
    report.set_item("dense_seconds", dense_seconds)?;
    report.set_item("hierarchical_seconds", hierarchical_seconds)?;
    report.set_item("speedup", dense_seconds / hierarchical_seconds)?;
    report.set_item("maximum_absolute_error", maximum_absolute_error)?;
    report.set_item("checksum_difference", checksum_difference)?;
    report.set_item("blocks", blocks.len())?;
    Ok(report.into())
}
