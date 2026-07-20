use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyTuple};

use crate::bidirectional::{search_bidirectional_top_k_all, BidirectionalTopKReport, TopKOptions};
use crate::errors::SamplerError;
use crate::input::{
    parse_contexts, parse_destination_inputs, parse_od_costs, parse_reference_contexts,
};
use crate::kernel_experiment::benchmark_hierarchical_kernel;
use crate::model::{DestinationIndex, OdGraph};
use crate::output::to_polars_dataframe;
use crate::particle::{sample_particles_all, ParticleReport};
use crate::profile::ProfileReport;
use crate::sampler::{sample_all, sample_all_with_profile, IterationCache, Parameters};
use crate::second_order::{solve_second_order_all, SecondOrderResult};
use crate::ternary_reference::{sample_reference_all, search_reference_top_k, HeapSearchReport};

/// Historical research surface retained for the experiment scripts.
///
/// New code should use `DestinationPlanSearch`, which exposes only the active
/// bounded top-K search and its exact oracle.
#[pyclass]
pub struct ExperimentalDestinationSampler {
    graph: OdGraph,
    destination_index: DestinationIndex,
    cache: IterationCache,
}

#[pymethods]
impl ExperimentalDestinationSampler {
    #[new]
    #[pyo3(signature = (*, od_costs, destination_inputs))]
    fn new(
        od_costs: &Bound<'_, PyAny>,
        destination_inputs: &Bound<'_, PyAny>,
    ) -> Result<Self, SamplerError> {
        let graph = OdGraph::build(parse_od_costs(od_costs)?)?;
        let destination_index =
            DestinationIndex::build(parse_destination_inputs(destination_inputs)?, &graph)?;
        Ok(Self {
            graph,
            destination_index,
            cache: IterationCache::default(),
        })
    }

    #[pyo3(signature = (*, steps, initial_locations, logit_scale, update_plan_timings, use_shadow_prices, seed, n_draws=1, n_threads=None, skip_infeasible=false))]
    #[allow(clippy::too_many_arguments)]
    fn sample(
        &self,
        py: Python<'_>,
        steps: &Bound<'_, PyAny>,
        initial_locations: &Bound<'_, PyAny>,
        logit_scale: f64,
        update_plan_timings: bool,
        use_shadow_prices: bool,
        seed: u64,
        n_draws: u32,
        n_threads: Option<usize>,
        skip_infeasible: bool,
    ) -> PyResult<PyObject> {
        validate_parameters(logit_scale, n_draws)?;
        let contexts = parse_contexts(steps, initial_locations)?;
        let parameters = Parameters {
            logit_scale,
            update_plan_timings,
            use_shadow_prices,
            seed,
            n_draws,
            skip_infeasible,
            wrapped_home_time_shadow_price: 0.0,
            use_bidirectional_feasibility: false,
        };
        let output = py.allow_threads(|| {
            sample_all(
                &self.graph,
                &self.destination_index,
                &self.cache,
                &contexts,
                parameters,
                n_threads,
            )
        })?;
        to_polars_dataframe(py, output)
    }

    /// Run the same sampler and return aggregate phase timings and workload
    /// counts. Profiling is explicit so normal model runs keep the lean path.
    #[pyo3(signature = (*, steps, initial_locations, logit_scale, update_plan_timings, use_shadow_prices, seed, n_draws=1, n_threads=None, skip_infeasible=false))]
    #[allow(clippy::too_many_arguments)]
    fn sample_with_profile(
        &self,
        py: Python<'_>,
        steps: &Bound<'_, PyAny>,
        initial_locations: &Bound<'_, PyAny>,
        logit_scale: f64,
        update_plan_timings: bool,
        use_shadow_prices: bool,
        seed: u64,
        n_draws: u32,
        n_threads: Option<usize>,
        skip_infeasible: bool,
    ) -> PyResult<PyObject> {
        validate_parameters(logit_scale, n_draws)?;
        let contexts = parse_contexts(steps, initial_locations)?;
        let parameters = Parameters {
            logit_scale,
            update_plan_timings,
            use_shadow_prices,
            seed,
            n_draws,
            skip_infeasible,
            wrapped_home_time_shadow_price: 0.0,
            use_bidirectional_feasibility: false,
        };
        let (output, profile) = py.allow_threads(|| {
            sample_all_with_profile(
                &self.graph,
                &self.destination_index,
                &self.cache,
                &contexts,
                parameters,
                n_threads,
            )
        })?;
        let output = to_polars_dataframe(py, output)?;
        let profile = profile_to_dict(py, &profile)?;
        Ok(PyTuple::new(py, [output, profile])?.into())
    }

    /// Enumerate complete destination assignments and score the exact
    /// rigidity-adjusted schedule. This is a small-domain reference, not a
    /// production search path.
    #[pyo3(signature = (*, steps, initial_locations, logit_scale, update_plan_timings, use_shadow_prices, seed, n_draws=1, max_assignments=100_000, skip_infeasible=false))]
    #[allow(clippy::too_many_arguments)]
    fn sample_ternary_reference(
        &self,
        py: Python<'_>,
        steps: &Bound<'_, PyAny>,
        initial_locations: &Bound<'_, PyAny>,
        logit_scale: f64,
        update_plan_timings: bool,
        use_shadow_prices: bool,
        seed: u64,
        n_draws: u32,
        max_assignments: usize,
        skip_infeasible: bool,
    ) -> PyResult<PyObject> {
        validate_parameters(logit_scale, n_draws)?;
        if max_assignments == 0 {
            return Err(
                SamplerError::InvalidInput("max_assignments must be positive".to_string()).into(),
            );
        }
        let contexts = parse_reference_contexts(steps, initial_locations)?;
        let parameters = Parameters {
            logit_scale,
            update_plan_timings,
            use_shadow_prices,
            seed,
            n_draws,
            skip_infeasible,
            wrapped_home_time_shadow_price: 0.0,
            use_bidirectional_feasibility: false,
        };
        let output = py.allow_threads(|| {
            sample_reference_all(
                &self.graph,
                &self.destination_index,
                &contexts,
                parameters,
                max_assignments,
            )
        })?;
        to_polars_dataframe(py, output)
    }

    /// Find the exact highest-utility complete plans with best-first search.
    /// The returned profile shows how much of the assignment lattice was
    /// explored before the top-k result was proven.
    #[pyo3(signature = (*, steps, initial_locations, logit_scale, update_plan_timings, use_shadow_prices, k=10, max_states=2_000_000, n_threads=None, skip_infeasible=false))]
    #[allow(clippy::too_many_arguments)]
    fn search_ternary_top_k(
        &self,
        py: Python<'_>,
        steps: &Bound<'_, PyAny>,
        initial_locations: &Bound<'_, PyAny>,
        logit_scale: f64,
        update_plan_timings: bool,
        use_shadow_prices: bool,
        k: usize,
        max_states: usize,
        n_threads: Option<usize>,
        skip_infeasible: bool,
    ) -> PyResult<PyObject> {
        validate_parameters(logit_scale, 1)?;
        if k == 0 || max_states == 0 {
            return Err(SamplerError::InvalidInput(
                "k and max_states must be positive".to_string(),
            )
            .into());
        }
        let contexts = parse_reference_contexts(steps, initial_locations)?;
        let parameters = Parameters {
            logit_scale,
            update_plan_timings,
            use_shadow_prices,
            seed: 0,
            n_draws: 1,
            skip_infeasible,
            wrapped_home_time_shadow_price: 0.0,
            use_bidirectional_feasibility: false,
        };
        let (output, report) = py.allow_threads(|| {
            search_reference_top_k(
                &self.graph,
                &self.destination_index,
                &contexts,
                parameters,
                k,
                max_states,
                n_threads,
            )
        })?;
        let output = to_polars_dataframe(py, output)?;
        let report = heap_report_to_dict(py, &report)?;
        Ok(PyTuple::new(py, [output, report])?.into())
    }

    /// Experimental rigidity-aware recursion used to validate aggregated
    /// destination models. It returns the partition and first marginal only.
    #[pyo3(signature = (*, steps, initial_locations, logit_scale, update_plan_timings, use_shadow_prices, wrapped_home_time_shadow_price=0.0, use_bidirectional_feasibility=true, n_threads=None, skip_infeasible=false))]
    #[allow(clippy::too_many_arguments)]
    fn solve_second_order(
        &self,
        py: Python<'_>,
        steps: &Bound<'_, PyAny>,
        initial_locations: &Bound<'_, PyAny>,
        logit_scale: f64,
        update_plan_timings: bool,
        use_shadow_prices: bool,
        wrapped_home_time_shadow_price: f64,
        use_bidirectional_feasibility: bool,
        n_threads: Option<usize>,
        skip_infeasible: bool,
    ) -> PyResult<PyObject> {
        validate_parameters(logit_scale, 1)?;
        if !wrapped_home_time_shadow_price.is_finite() || wrapped_home_time_shadow_price < 0.0 {
            return Err(SamplerError::InvalidInput(
                "wrapped_home_time_shadow_price must be finite and non-negative".to_string(),
            )
            .into());
        }
        let contexts = parse_reference_contexts(steps, initial_locations)?;
        let parameters = Parameters {
            logit_scale,
            update_plan_timings,
            use_shadow_prices,
            seed: 0,
            n_draws: 1,
            skip_infeasible,
            wrapped_home_time_shadow_price,
            use_bidirectional_feasibility,
        };
        let result = py.allow_threads(|| {
            solve_second_order_all(
                &self.graph,
                &self.destination_index,
                &contexts,
                parameters,
                n_threads,
            )
        })?;
        second_order_to_dict(py, result)
    }

    /// Bounded sequential particle proposal sampler. Complete particles are
    /// rescored with the rigidity-aware reference scorer; its report exposes
    /// candidate work and importance-weight ESS for small-case validation.
    #[pyo3(signature = (*, steps, initial_locations, logit_scale, update_plan_timings, use_shadow_prices, seed, n_particles=32, candidate_count=16, max_retries=2, n_draws=1, n_threads=None, skip_infeasible=false))]
    #[allow(clippy::too_many_arguments)]
    fn sample_particles(
        &self,
        py: Python<'_>,
        steps: &Bound<'_, PyAny>,
        initial_locations: &Bound<'_, PyAny>,
        logit_scale: f64,
        update_plan_timings: bool,
        use_shadow_prices: bool,
        seed: u64,
        n_particles: usize,
        candidate_count: usize,
        max_retries: u32,
        n_draws: u32,
        n_threads: Option<usize>,
        skip_infeasible: bool,
    ) -> PyResult<PyObject> {
        validate_parameters(logit_scale, n_draws)?;
        if n_particles == 0 || candidate_count == 0 || n_draws as usize > n_particles {
            return Err(SamplerError::InvalidInput(
                "n_particles, candidate_count, and n_draws must be positive, and n_draws cannot exceed n_particles".to_string(),
            ).into());
        }
        let contexts = parse_reference_contexts(steps, initial_locations)?;
        let parameters = Parameters {
            logit_scale,
            update_plan_timings,
            use_shadow_prices,
            seed,
            n_draws,
            skip_infeasible,
            wrapped_home_time_shadow_price: 0.0,
            use_bidirectional_feasibility: false,
        };
        let (output, report) = py.allow_threads(|| {
            sample_particles_all(
                &self.graph,
                &self.destination_index,
                &contexts,
                parameters,
                n_particles,
                candidate_count,
                max_retries,
                n_threads,
            )
        })?;
        Ok(PyTuple::new(
            py,
            [
                to_polars_dataframe(py, output)?,
                particle_report_to_dict(py, &report)?,
            ],
        )?
        .into())
    }

    /// Experimental bounded stitch-layer top-K search. It keeps deterministic
    /// beam frontiers from both homes, then stitches the two remaining
    /// stitch-boundary factors and returns the exact-score-ranked plans. Repeated
    /// variable anchors are constrained to one common destination.
    #[pyo3(signature = (*, steps, initial_locations, logit_scale, update_plan_timings, use_shadow_prices, seed, beam_width=32, candidate_count=16, top_k=1, n_threads=None, skip_infeasible=false, profile=false))]
    #[allow(clippy::too_many_arguments)]
    fn search_bidirectional_top_k(
        &self,
        py: Python<'_>,
        steps: &Bound<'_, PyAny>,
        initial_locations: &Bound<'_, PyAny>,
        logit_scale: f64,
        update_plan_timings: bool,
        use_shadow_prices: bool,
        seed: u64,
        beam_width: usize,
        candidate_count: usize,
        top_k: u32,
        n_threads: Option<usize>,
        skip_infeasible: bool,
        profile: bool,
    ) -> PyResult<PyObject> {
        validate_parameters(logit_scale, top_k)?;
        if beam_width == 0 || candidate_count == 0 {
            return Err(SamplerError::InvalidInput(
                "beam_width and candidate_count must be positive".to_string(),
            )
            .into());
        }
        let contexts = parse_reference_contexts(steps, initial_locations)?;
        let parameters = Parameters {
            logit_scale,
            update_plan_timings,
            use_shadow_prices,
            seed,
            n_draws: top_k,
            skip_infeasible,
            wrapped_home_time_shadow_price: 0.0,
            use_bidirectional_feasibility: false,
        };
        let (output, report) = py.allow_threads(|| {
            search_bidirectional_top_k_all(
                &self.graph,
                &self.destination_index,
                &contexts,
                parameters,
                TopKOptions {
                    frontier_width: beam_width,
                    proposal_limit_per_source: candidate_count,
                    stitch_bias: 0,
                    continuation_state_limit: 4,
                    continuation_proposal_limit: 4,
                    seam_refresh_per_prefix: 0,
                    profile,
                },
                n_threads,
            )
        })?;
        Ok(PyTuple::new(
            py,
            [
                to_polars_dataframe(py, output)?,
                bidirectional_top_k_report_to_dict(py, &report)?,
            ],
        )?
        .into())
    }
}

/// Active deterministic destination-plan search.
///
/// The search owns bounded destination proposals, forward/backward frontiers,
/// exact-score stitching, and final top-K ranking. It deliberately does not
/// expose the historical sampling and aggregate-solver experiments.
#[pyclass]
pub struct DestinationPlanSearch {
    graph: OdGraph,
    destination_index: DestinationIndex,
}

#[pymethods]
impl DestinationPlanSearch {
    #[new]
    #[pyo3(signature = (*, od_costs, destination_inputs))]
    fn new(
        od_costs: &Bound<'_, PyAny>,
        destination_inputs: &Bound<'_, PyAny>,
    ) -> Result<Self, SamplerError> {
        let graph = OdGraph::build(parse_od_costs(od_costs)?)?;
        let destination_index =
            DestinationIndex::build(parse_destination_inputs(destination_inputs)?, &graph)?;
        Ok(Self {
            graph,
            destination_index,
        })
    }

    /// Return the bounded, exact-score-ranked destination plans.
    #[pyo3(signature = (*, steps, initial_locations, logit_scale, update_plan_timings, use_shadow_prices, exploration_seed, frontier_width=32, proposal_limit_per_source=16, stitch_bias=0, continuation_state_limit=1, continuation_proposal_limit=1, seam_refresh_per_prefix=1, top_k=10, n_threads=None, skip_infeasible=false, collect_profile=false))]
    #[allow(clippy::too_many_arguments)]
    fn top_k(
        &self,
        py: Python<'_>,
        steps: &Bound<'_, PyAny>,
        initial_locations: &Bound<'_, PyAny>,
        logit_scale: f64,
        update_plan_timings: bool,
        use_shadow_prices: bool,
        exploration_seed: u64,
        frontier_width: usize,
        proposal_limit_per_source: usize,
        stitch_bias: i32,
        continuation_state_limit: usize,
        continuation_proposal_limit: usize,
        seam_refresh_per_prefix: usize,
        top_k: u32,
        n_threads: Option<usize>,
        skip_infeasible: bool,
        collect_profile: bool,
    ) -> PyResult<PyObject> {
        validate_parameters(logit_scale, top_k)?;
        if frontier_width == 0
            || proposal_limit_per_source == 0
            || continuation_state_limit == 0
            || continuation_proposal_limit == 0
        {
            return Err(SamplerError::InvalidInput(
                "frontier_width, proposal_limit_per_source, continuation_state_limit, and continuation_proposal_limit must be positive".to_string(),
            )
            .into());
        }
        let contexts = parse_reference_contexts(steps, initial_locations)?;
        let parameters = Parameters {
            logit_scale,
            update_plan_timings,
            use_shadow_prices,
            seed: exploration_seed,
            n_draws: top_k,
            skip_infeasible,
            wrapped_home_time_shadow_price: 0.0,
            use_bidirectional_feasibility: false,
        };
        let (output, report) = py.allow_threads(|| {
            search_bidirectional_top_k_all(
                &self.graph,
                &self.destination_index,
                &contexts,
                parameters,
                TopKOptions {
                    frontier_width,
                    proposal_limit_per_source,
                    stitch_bias,
                    continuation_state_limit,
                    continuation_proposal_limit,
                    seam_refresh_per_prefix,
                    profile: collect_profile,
                },
                n_threads,
            )
        })?;
        Ok(PyTuple::new(
            py,
            [
                to_polars_dataframe(py, output)?,
                top_k_report_to_dict(py, &report)?,
            ],
        )?
        .into())
    }

    /// Prove the highest-utility complete plans on a bounded exact workload.
    #[pyo3(signature = (*, steps, initial_locations, logit_scale, update_plan_timings, use_shadow_prices, top_k=10, max_states=2_000_000, n_threads=None, skip_infeasible=false))]
    #[allow(clippy::too_many_arguments)]
    fn exact_top_k(
        &self,
        py: Python<'_>,
        steps: &Bound<'_, PyAny>,
        initial_locations: &Bound<'_, PyAny>,
        logit_scale: f64,
        update_plan_timings: bool,
        use_shadow_prices: bool,
        top_k: usize,
        max_states: usize,
        n_threads: Option<usize>,
        skip_infeasible: bool,
    ) -> PyResult<PyObject> {
        validate_parameters(logit_scale, 1)?;
        if top_k == 0 || max_states == 0 {
            return Err(SamplerError::InvalidInput(
                "top_k and max_states must be positive".to_string(),
            )
            .into());
        }
        let contexts = parse_reference_contexts(steps, initial_locations)?;
        let parameters = Parameters {
            logit_scale,
            update_plan_timings,
            use_shadow_prices,
            seed: 0,
            n_draws: 1,
            skip_infeasible,
            wrapped_home_time_shadow_price: 0.0,
            use_bidirectional_feasibility: false,
        };
        let (output, report) = py.allow_threads(|| {
            search_reference_top_k(
                &self.graph,
                &self.destination_index,
                &contexts,
                parameters,
                top_k,
                max_states,
                n_threads,
            )
        })?;
        Ok(PyTuple::new(
            py,
            [
                to_polars_dataframe(py, output)?,
                heap_report_to_dict(py, &report)?,
            ],
        )?
        .into())
    }
}

fn top_k_report_to_dict(py: Python<'_>, report: &BidirectionalTopKReport) -> PyResult<PyObject> {
    let result = PyDict::new(py);
    result.set_item("contexts", report.contexts)?;
    result.set_item(
        "forward_proposals_evaluated",
        report.forward_candidate_evaluations,
    )?;
    result.set_item(
        "backward_proposals_evaluated",
        report.backward_candidate_evaluations,
    )?;
    result.set_item("continuation_proposals", report.continuation_proposals)?;
    result.set_item("seam_refresh_proposals", report.seam_refresh_proposals)?;
    result.set_item("seam_refresh_states", report.seam_refresh_states)?;
    result.set_item("stitch_pairs", report.stitch_pairs)?;
    result.set_item("complete_plan_candidates", report.completed_plans)?;
    result.set_item("infeasible_contexts", report.infeasible_contexts)?;
    result.set_item("build_problem_ns", report.build_problem_ns)?;
    result.set_item("backward_search_ns", report.backward_search_ns)?;
    result.set_item("backward_guidance_ns", report.backward_guidance_ns)?;
    result.set_item("forward_search_ns", report.forward_search_ns)?;
    result.set_item("continuation_guidance_ns", report.continuation_guidance_ns)?;
    result.set_item("seam_refresh_ns", report.seam_refresh_ns)?;
    result.set_item("stitch_ns", report.stitch_ns)?;
    result.set_item("materialize_ns", report.materialize_ns)?;
    result.set_item("total_search_ns", report.total_search_ns)?;
    Ok(result.into())
}

fn bidirectional_top_k_report_to_dict(
    py: Python<'_>,
    report: &BidirectionalTopKReport,
) -> PyResult<PyObject> {
    let result = PyDict::new(py);
    result.set_item("contexts", report.contexts)?;
    result.set_item(
        "forward_candidate_evaluations",
        report.forward_candidate_evaluations,
    )?;
    result.set_item(
        "backward_candidate_evaluations",
        report.backward_candidate_evaluations,
    )?;
    result.set_item("continuation_proposals", report.continuation_proposals)?;
    result.set_item("seam_refresh_proposals", report.seam_refresh_proposals)?;
    result.set_item("seam_refresh_states", report.seam_refresh_states)?;
    result.set_item("stitch_pairs", report.stitch_pairs)?;
    result.set_item("completed_plans", report.completed_plans)?;
    result.set_item("infeasible_contexts", report.infeasible_contexts)?;
    result.set_item("build_problem_ns", report.build_problem_ns)?;
    result.set_item("backward_search_ns", report.backward_search_ns)?;
    result.set_item("backward_guidance_ns", report.backward_guidance_ns)?;
    result.set_item("forward_search_ns", report.forward_search_ns)?;
    result.set_item("continuation_guidance_ns", report.continuation_guidance_ns)?;
    result.set_item("seam_refresh_ns", report.seam_refresh_ns)?;
    result.set_item("stitch_ns", report.stitch_ns)?;
    result.set_item("materialize_ns", report.materialize_ns)?;
    result.set_item("total_search_ns", report.total_search_ns)?;
    Ok(result.into())
}

fn particle_report_to_dict(py: Python<'_>, report: &ParticleReport) -> PyResult<PyObject> {
    let result = PyDict::new(py);
    result.set_item("contexts", report.contexts)?;
    result.set_item("candidate_evaluations", report.candidate_evaluations)?;
    result.set_item(
        "locally_infeasible_candidates",
        report.locally_infeasible_candidates,
    )?;
    result.set_item("completed_particles", report.completed_particles)?;
    result.set_item("selected_plans", report.selected_plans)?;
    result.set_item("infeasible_contexts", report.infeasible_contexts)?;
    result.set_item("retry_attempts", report.retry_attempts)?;
    result.set_item("recovered_contexts", report.recovered_contexts)?;
    result.set_item("effective_sample_size", report.effective_sample_size_sum)?;
    result.set_item(
        "mean_effective_sample_size",
        if report.contexts == 0 {
            0.0
        } else {
            report.effective_sample_size_sum / report.contexts as f64
        },
    )?;
    let context_reports = PyList::empty(py);
    for context in &report.context_reports {
        let item = PyDict::new(py);
        item.set_item("context_id", context.context_id)?;
        item.set_item("candidate_evaluations", context.candidate_evaluations)?;
        item.set_item(
            "locally_infeasible_candidates",
            context.locally_infeasible_candidates,
        )?;
        item.set_item("completed_particles", context.completed_particles)?;
        item.set_item("selected_plans", context.selected_plans)?;
        item.set_item("effective_sample_size", context.effective_sample_size)?;
        item.set_item("first_failure_layer", context.first_failure_layer)?;
        item.set_item("failure_reason", context.failure_reason)?;
        item.set_item(
            "candidate_set_size_at_failure",
            context.candidate_set_size_at_failure,
        )?;
        item.set_item(
            "domain_locally_feasible_candidates",
            context.domain_locally_feasible_candidates,
        )?;
        item.set_item("retry_attempts", context.retry_attempts)?;
        item.set_item("recovered_by_retry", context.recovered_by_retry)?;
        context_reports.append(item)?;
    }
    result.set_item("context_reports", context_reports)?;
    Ok(result.into())
}

fn second_order_to_dict(py: Python<'_>, result: SecondOrderResult) -> PyResult<PyObject> {
    let output = PyDict::new(py);
    output.set_item("context_ids", result.context_ids)?;
    output.set_item("log_partitions", result.log_partitions)?;
    output.set_item(
        "first_destination_probabilities",
        result.first_destination_probabilities,
    )?;
    output.set_item("zone_ids", result.zone_ids)?;
    output.set_item("wall_seconds", result.wall_time.as_secs_f64())?;
    output.set_item("anchor_conditions", result.anchor_conditions)?;
    output.set_item("infeasible_contexts", result.infeasible_contexts)?;
    output.set_item("duration_checks", result.duration_checks)?;
    output.set_item("duration_infeasible", result.duration_infeasible)?;
    output.set_item("scored_transitions", result.scored_transitions)?;
    output.set_item("pair_states", result.pair_states)?;
    output.set_item("feasible_pair_states", result.feasible_pair_states)?;
    output.set_item("forward_pair_states", result.forward_pair_states)?;
    output.set_item(
        "forward_reachable_pair_states",
        result.forward_reachable_pair_states,
    )?;
    output.set_item("forward_time_edge_scans", result.forward_time_edge_scans)?;
    output.set_item("forward_time_cutoffs", result.forward_time_cutoffs)?;
    output.set_item("backward_time_edge_scans", result.backward_time_edge_scans)?;
    output.set_item("backward_time_cutoffs", result.backward_time_cutoffs)?;
    output.set_item("corridor_pair_states", result.corridor_pair_states)?;
    Ok(output.into())
}

fn heap_report_to_dict(py: Python<'_>, report: &HeapSearchReport) -> PyResult<PyObject> {
    let result = PyDict::new(py);
    result.set_item("contexts", report.contexts)?;
    result.set_item("split_contexts", report.split_contexts)?;
    result.set_item(
        "conditioned_anchor_contexts",
        report.conditioned_anchor_contexts,
    )?;
    result.set_item(
        "anchor_conditions_considered",
        report.anchor_conditions_considered,
    )?;
    result.set_item("anchor_conditions_pruned", report.anchor_conditions_pruned)?;
    result.set_item("incumbent_contexts", report.incumbent_contexts)?;
    result.set_item(
        "incumbent_children_considered",
        report.incumbent_children_considered,
    )?;
    result.set_item(
        "children_pruned_by_incumbent",
        report.children_pruned_by_incumbent,
    )?;
    result.set_item("queue_entries_popped", report.queue_entries_popped)?;
    result.set_item("sibling_entries_popped", report.sibling_entries_popped)?;
    result.set_item("states_popped", report.states_popped)?;
    result.set_item("states_pushed", report.states_pushed)?;
    result.set_item("children_considered", report.children_considered)?;
    result.set_item("complete_plans", report.complete_plans)?;
    result.set_item("maximum_heap_size", report.maximum_heap_size)?;
    result.set_item("assignment_lattice", report.assignment_lattice.to_string())?;
    Ok(result.into())
}

fn profile_to_dict(py: Python<'_>, profile: &ProfileReport) -> PyResult<PyObject> {
    let result = PyDict::new(py);
    result.set_item("plan_build_seconds", profile.plan_build.as_secs_f64())?;
    result.set_item("sampling_wall_seconds", profile.sampling_wall.as_secs_f64())?;
    result.set_item("output_merge_seconds", profile.output_merge.as_secs_f64())?;
    result.set_item("context_cpu_seconds", profile.context_cpu.as_secs_f64())?;
    result.set_item("anchor_cpu_seconds", profile.anchor_cpu.as_secs_f64())?;
    result.set_item("tree_cpu_seconds", profile.tree_cpu.as_secs_f64())?;
    result.set_item(
        "tree_problem_build_cpu_seconds",
        profile.tree_problem_build_cpu.as_secs_f64(),
    )?;
    result.set_item(
        "tree_structure_build_cpu_seconds",
        profile.tree_structure_build_cpu.as_secs_f64(),
    )?;
    result.set_item(
        "tree_backward_cpu_seconds",
        profile.tree_backward_cpu.as_secs_f64(),
    )?;
    result.set_item(
        "tree_forward_cpu_seconds",
        profile.tree_forward_cpu.as_secs_f64(),
    )?;
    result.set_item("contexts", profile.contexts)?;
    result.set_item("anchor_contexts", profile.anchor_contexts)?;
    result.set_item("tree_contexts", profile.tree_contexts)?;
    result.set_item("successful_contexts", profile.successful_contexts)?;
    result.set_item("infeasible_contexts", profile.infeasible_contexts)?;
    result.set_item("cyclic_contexts", profile.cyclic_contexts)?;
    result.set_item("input_steps", profile.input_steps)?;
    result.set_item("output_rows", profile.output_rows)?;
    result.set_item("variables", profile.variables)?;
    result.set_item("pair_factors", profile.pair_factors)?;
    result.set_item("pair_transitions", profile.pair_transitions)?;
    result.set_item("domain_choices", profile.domain_choices)?;
    result.set_item("outgoing_messages", profile.outgoing_messages)?;
    result.set_item("message_edges", profile.message_edges)?;
    Ok(result.into())
}

fn validate_parameters(logit_scale: f64, n_draws: u32) -> Result<(), SamplerError> {
    if !logit_scale.is_finite() || logit_scale <= 0.0 {
        return Err(SamplerError::InvalidInput(
            "logit_scale must be finite and positive".to_string(),
        ));
    }
    if n_draws == 0 {
        return Err(SamplerError::InvalidInput(
            "n_draws must be positive".to_string(),
        ));
    }
    Ok(())
}

/// Compute exact backward destination values and sample complete chains forward.
#[pyfunction]
#[pyo3(signature = (*, steps, initial_locations, od_costs, destination_inputs, logit_scale, update_plan_timings, use_shadow_prices, seed, n_draws=1, n_threads=None, skip_infeasible=false))]
#[allow(clippy::too_many_arguments)]
pub fn sample_destination_sequences(
    py: Python<'_>,
    steps: &Bound<'_, PyAny>,
    initial_locations: &Bound<'_, PyAny>,
    od_costs: &Bound<'_, PyAny>,
    destination_inputs: &Bound<'_, PyAny>,
    logit_scale: f64,
    update_plan_timings: bool,
    use_shadow_prices: bool,
    seed: u64,
    n_draws: u32,
    n_threads: Option<usize>,
    skip_infeasible: bool,
) -> PyResult<PyObject> {
    validate_parameters(logit_scale, n_draws)?;

    // Python prepares four compact tables. Rust converts them once, then all
    // backward and forward work runs without DataFrame joins or Python calls.
    let graph = OdGraph::build(parse_od_costs(od_costs)?)?;
    let destination_index =
        DestinationIndex::build(parse_destination_inputs(destination_inputs)?, &graph)?;
    let contexts = parse_contexts(steps, initial_locations)?;
    let parameters = Parameters {
        logit_scale,
        update_plan_timings,
        use_shadow_prices,
        seed,
        n_draws,
        skip_infeasible,
        wrapped_home_time_shadow_price: 0.0,
        use_bidirectional_feasibility: false,
    };
    let cache = IterationCache::default();
    let output = py.allow_threads(|| {
        sample_all(
            &graph,
            &destination_index,
            &cache,
            &contexts,
            parameters,
            n_threads,
        )
    })?;
    to_polars_dataframe(py, output)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<DestinationPlanSearch>()?;
    module.add_class::<ExperimentalDestinationSampler>()?;
    module.add_function(wrap_pyfunction!(sample_destination_sequences, module)?)?;
    module.add_function(wrap_pyfunction!(benchmark_hierarchical_kernel, module)?)?;
    Ok(())
}
