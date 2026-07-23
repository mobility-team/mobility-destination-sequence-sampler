use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyTuple};

use crate::errors::SamplerError;
use crate::input::{parse_destination_inputs, parse_od_costs, parse_reference_contexts};
use crate::model::{DestinationIndex, OdGraph};
use crate::oracle::{enumerate_reference_distribution, search_reference_top_k, HeapSearchReport};
use crate::output::to_polars_dataframe;
use crate::scoring::Parameters;
use crate::top_k::{
    search_top_k_all, ActiveTraceRequest, CandidateStrategy, TopKOptions, TopKReport,
};
use std::sync::Arc;

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
    #[pyo3(signature = (*, steps, initial_locations, logit_scale, update_plan_timings, use_shadow_prices, exploration_seed, frontier_width=40, proposal_limit_per_source=16, symmetric_message_limit=4, symmetric_state_limit=4, symmetric_forward_proposal_limit=20, candidate_strategy="symmetric_factor_map", surface_bins=2, factor_map_max_depth=99, stitch_bias=1, continuation_state_limit=1, deep_continuation_state_limit=16, continuation_log_gap=0.0, continuation_proposal_limit=1, seam_refresh_per_prefix=1, heuristic_reserve_limit=0, top_k=10, n_threads=None, skip_infeasible=false, collect_profile=false, active_trace_context_id=None, active_trace_target_plans=None))]
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
        symmetric_message_limit: usize,
        symmetric_state_limit: usize,
        symmetric_forward_proposal_limit: usize,
        candidate_strategy: &str,
        surface_bins: usize,
        factor_map_max_depth: usize,
        stitch_bias: i32,
        continuation_state_limit: usize,
        deep_continuation_state_limit: usize,
        continuation_log_gap: f64,
        continuation_proposal_limit: usize,
        seam_refresh_per_prefix: usize,
        heuristic_reserve_limit: usize,
        top_k: u32,
        n_threads: Option<usize>,
        skip_infeasible: bool,
        collect_profile: bool,
        active_trace_context_id: Option<u64>,
        active_trace_target_plans: Option<Vec<Vec<u32>>>,
    ) -> PyResult<PyObject> {
        validate_logit_scale(logit_scale)?;
        validate_top_k(top_k as usize)?;
        if frontier_width == 0
            || proposal_limit_per_source == 0
            || continuation_state_limit == 0
            || deep_continuation_state_limit == 0
            || continuation_proposal_limit == 0
        {
            return Err(SamplerError::InvalidInput(
                "frontier_width, proposal_limit_per_source, continuation state limits, and continuation_proposal_limit must be positive".to_string(),
            )
            .into());
        }
        if !matches!(surface_bins, 2 | 4) {
            return Err(
                SamplerError::InvalidInput("surface_bins must be 2 or 4".to_string()).into(),
            );
        }
        if !continuation_log_gap.is_finite() || continuation_log_gap < 0.0 {
            return Err(SamplerError::InvalidInput(
                "continuation_log_gap must be finite and non-negative".to_string(),
            )
            .into());
        }
        if factor_map_max_depth < 2 {
            return Err(SamplerError::InvalidInput(
                "factor_map_max_depth must be at least 2".to_string(),
            )
            .into());
        }
        let contexts = parse_reference_contexts(steps, initial_locations)?;
        let candidate_strategy = CandidateStrategy::parse(candidate_strategy)?;
        let active_trace = match (active_trace_context_id, active_trace_target_plans) {
            (None, None) => None,
            (Some(context_id), Some(target_plans)) => Some(ActiveTraceRequest {
                context_id,
                target_plans: Arc::from(target_plans),
            }),
            _ => return Err(SamplerError::InvalidInput(
                "active_trace_context_id and active_trace_target_plans must be supplied together"
                    .to_string(),
            )
            .into()),
        };
        let parameters = Parameters {
            logit_scale,
            update_plan_timings,
            use_shadow_prices,
            skip_infeasible,
        };
        let (output, report) = py.allow_threads(|| {
            search_top_k_all(
                &self.graph,
                &self.destination_index,
                &contexts,
                parameters,
                TopKOptions {
                    exploration_seed,
                    result_limit: top_k,
                    frontier_width,
                    proposal_limit_per_source,
                    symmetric_message_limit,
                    symmetric_state_limit,
                    symmetric_forward_proposal_limit,
                    candidate_strategy,
                    surface_bins,
                    factor_map_max_depth,
                    stitch_bias,
                    continuation_state_limit,
                    deep_continuation_state_limit,
                    continuation_log_gap,
                    continuation_proposal_limit,
                    seam_refresh_per_prefix,
                    heuristic_reserve_limit,
                    profile: collect_profile,
                    active_trace,
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
        validate_logit_scale(logit_scale)?;
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
            skip_infeasible,
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

    /// Enumerate the full exact exp(U) distribution for one small context.
    #[pyo3(signature = (*, steps, initial_locations, logit_scale, update_plan_timings, use_shadow_prices, max_assignments=100_000))]
    #[allow(clippy::too_many_arguments)]
    fn exact_distribution(
        &self,
        py: Python<'_>,
        steps: &Bound<'_, PyAny>,
        initial_locations: &Bound<'_, PyAny>,
        logit_scale: f64,
        update_plan_timings: bool,
        use_shadow_prices: bool,
        max_assignments: usize,
    ) -> PyResult<PyObject> {
        validate_logit_scale(logit_scale)?;
        if max_assignments == 0 {
            return Err(
                SamplerError::InvalidInput("max_assignments must be positive".to_string()).into(),
            );
        }
        let contexts = parse_reference_contexts(steps, initial_locations)?;
        if contexts.len() != 1 {
            return Err(SamplerError::InvalidInput(
                "exact_distribution accepts exactly one context".to_string(),
            )
            .into());
        }
        let parameters = Parameters {
            logit_scale,
            update_plan_timings,
            use_shadow_prices,
            skip_infeasible: false,
        };
        let distribution = py.allow_threads(|| {
            enumerate_reference_distribution(
                &self.graph,
                &self.destination_index,
                &contexts[0],
                parameters,
                max_assignments,
            )
        })?;
        let maximum = distribution.scores[0];
        let log_normalizer = maximum
            + distribution
                .scores
                .iter()
                .map(|score| (score - maximum).exp())
                .sum::<f64>()
                .ln();
        let feasible_plans = distribution.scores.len();
        let result = PyDict::new(py);
        result.set_item("scores", distribution.scores)?;
        result.set_item("feasible_plans", feasible_plans)?;
        result.set_item(
            "assignment_lattice",
            distribution.assignment_lattice.to_string(),
        )?;
        result.set_item("log_normalizer", log_normalizer)?;
        Ok(result.into())
    }
}

fn top_k_report_to_dict(py: Python<'_>, report: &TopKReport) -> PyResult<PyObject> {
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
    result.set_item(
        "surface_proposals_evaluated",
        report.surface_proposal_evaluations,
    )?;
    result.set_item(
        "factor_map_destinations_evaluated",
        report.factor_map_destination_evaluations,
    )?;
    result.set_item("factor_map_previous_hits", report.factor_map_previous_hits)?;
    result.set_item(
        "factor_map_previous_builds",
        report.factor_map_previous_builds,
    )?;
    result.set_item("factor_map_current_hits", report.factor_map_current_hits)?;
    result.set_item(
        "factor_map_current_builds",
        report.factor_map_current_builds,
    )?;
    result.set_item("factor_map_next_hits", report.factor_map_next_hits)?;
    result.set_item("factor_map_next_builds", report.factor_map_next_builds)?;
    result.set_item(
        "factor_map_previous_destination_scans",
        report.factor_map_previous_destination_scans,
    )?;
    result.set_item(
        "factor_map_current_destination_scans",
        report.factor_map_current_destination_scans,
    )?;
    result.set_item(
        "factor_map_next_destination_scans",
        report.factor_map_next_destination_scans,
    )?;
    result.set_item(
        "factor_map_previous_feasible_entries",
        report.factor_map_previous_feasible_entries,
    )?;
    result.set_item(
        "factor_map_current_feasible_entries",
        report.factor_map_current_feasible_entries,
    )?;
    result.set_item(
        "factor_map_next_feasible_entries",
        report.factor_map_next_feasible_entries,
    )?;
    result.set_item(
        "reverse_prefix_partial_calls",
        report.reverse_prefix_partial_calls,
    )?;
    result.set_item("local_score_cache_hits", report.local_score_cache_hits)?;
    result.set_item("local_score_cache_builds", report.local_score_cache_builds)?;
    result.set_item("continuation_proposals", report.continuation_proposals)?;
    result.set_item(
        "heuristic_reserve_triggers",
        report.heuristic_reserve_triggers,
    )?;
    result.set_item(
        "heuristic_reserve_proposals",
        report.heuristic_reserve_proposals,
    )?;
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
    result.set_item("surface_proposal_ns", report.surface_proposal_ns)?;
    result.set_item("factor_map_ns", report.factor_map_ns)?;
    result.set_item("seam_refresh_ns", report.seam_refresh_ns)?;
    result.set_item("stitch_ns", report.stitch_ns)?;
    result.set_item("materialize_ns", report.materialize_ns)?;
    result.set_item("total_search_ns", report.total_search_ns)?;
    let active_trace = report
        .active_trace_targets
        .iter()
        .map(|target| {
            let item = PyDict::new(py);
            item.set_item("zones", &target.zones)?;
            item.set_item("proposed", &target.proposed)?;
            item.set_item("retained", &target.retained)?;
            item.set_item("prefix_proposed", &target.prefix_proposed)?;
            item.set_item("prefix_retained", &target.prefix_retained)?;
            item.set_item("guidance_retained", &target.guidance_retained)?;
            item.set_item("guidance_proposed", &target.guidance_proposed)?;
            item.set_item("exact_guidance_rank", &target.exact_guidance_rank)?;
            item.set_item("exact_guidance_log_gap", &target.exact_guidance_log_gap)?;
            item.set_item(
                "prefix_pruned",
                target
                    .prefix_proposed
                    .iter()
                    .zip(&target.prefix_retained)
                    .map(|(proposed, retained)| match (proposed, retained) {
                        (Some(proposed), Some(retained)) => Some(*proposed && !*retained),
                        _ => None,
                    })
                    .collect::<Vec<_>>(),
            )?;
            item.set_item(
                "pruned",
                target
                    .proposed
                    .iter()
                    .zip(&target.retained)
                    .map(|(proposed, retained)| *proposed && !retained)
                    .collect::<Vec<_>>(),
            )?;
            Ok(item.into_any().unbind())
        })
        .collect::<PyResult<Vec<_>>>()?;
    result.set_item("active_trace_targets", active_trace)?;
    Ok(result.into())
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

fn validate_logit_scale(logit_scale: f64) -> Result<(), SamplerError> {
    if !logit_scale.is_finite() || logit_scale <= 0.0 {
        return Err(SamplerError::InvalidInput(
            "logit_scale must be finite and positive".to_string(),
        ));
    }
    Ok(())
}

fn validate_top_k(top_k: usize) -> Result<(), SamplerError> {
    if top_k == 0 {
        return Err(SamplerError::InvalidInput(
            "top_k must be positive".to_string(),
        ));
    }
    Ok(())
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<DestinationPlanSearch>()?;
    Ok(())
}
