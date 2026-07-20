use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyTuple};

use crate::errors::SamplerError;
use crate::input::{parse_destination_inputs, parse_od_costs, parse_reference_contexts};
use crate::model::{DestinationIndex, OdGraph};
use crate::oracle::{search_reference_top_k, HeapSearchReport};
use crate::output::to_polars_dataframe;
use crate::scoring::Parameters;
use crate::top_k::{search_top_k_all, TopKOptions, TopKReport};

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
        validate_logit_scale(logit_scale)?;
        validate_top_k(top_k as usize)?;
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
