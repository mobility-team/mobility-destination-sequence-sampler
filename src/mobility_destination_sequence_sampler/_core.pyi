from __future__ import annotations

from typing import Any, Literal, TypedDict


CandidateStrategy = Literal[
    "heuristic", "surface", "factor_map", "symmetric_factor_map"
]


class TopKReport(TypedDict):
    contexts: int
    forward_proposals_evaluated: int
    backward_proposals_evaluated: int
    surface_proposals_evaluated: int
    factor_map_destinations_evaluated: int
    factor_map_previous_hits: int
    factor_map_previous_builds: int
    factor_map_current_hits: int
    factor_map_current_builds: int
    factor_map_next_hits: int
    factor_map_next_builds: int
    factor_map_previous_destination_scans: int
    factor_map_current_destination_scans: int
    factor_map_next_destination_scans: int
    factor_map_previous_feasible_entries: int
    factor_map_current_feasible_entries: int
    factor_map_next_feasible_entries: int
    reverse_prefix_partial_calls: int
    local_score_cache_hits: int
    local_score_cache_builds: int
    continuation_proposals: int
    seam_refresh_proposals: int
    seam_refresh_states: int
    stitch_pairs: int
    complete_plan_candidates: int
    infeasible_contexts: int
    build_problem_ns: int
    backward_search_ns: int
    backward_guidance_ns: int
    forward_search_ns: int
    continuation_guidance_ns: int
    surface_proposal_ns: int
    factor_map_ns: int
    seam_refresh_ns: int
    stitch_ns: int
    materialize_ns: int
    total_search_ns: int


class ExactTopKReport(TypedDict):
    contexts: int
    split_contexts: int
    conditioned_anchor_contexts: int
    anchor_conditions_considered: int
    anchor_conditions_pruned: int
    incumbent_contexts: int
    incumbent_children_considered: int
    children_pruned_by_incumbent: int
    queue_entries_popped: int
    sibling_entries_popped: int
    states_popped: int
    states_pushed: int
    children_considered: int
    complete_plans: int
    maximum_heap_size: int
    assignment_lattice: str


class DestinationPlanSearch:
    def __init__(self, *, od_costs: Any, destination_inputs: Any) -> None: ...

    def top_k(
        self,
        *,
        steps: Any,
        initial_locations: Any,
        logit_scale: float,
        update_plan_timings: bool,
        use_shadow_prices: bool,
        exploration_seed: int,
        frontier_width: int = 40,
        proposal_limit_per_source: int = 16,
        symmetric_message_limit: int = 4,
        symmetric_state_limit: int = 4,
        symmetric_forward_proposal_limit: int = 8,
        candidate_strategy: CandidateStrategy = "symmetric_factor_map",
        surface_bins: int = 2,
        factor_map_max_depth: int = 5,
        stitch_bias: int = 1,
        continuation_state_limit: int = 1,
        continuation_proposal_limit: int = 1,
        seam_refresh_per_prefix: int = 1,
        heuristic_reserve_limit: int = 0,
        top_k: int = 10,
        n_threads: int | None = None,
        skip_infeasible: bool = False,
        collect_profile: bool = False,
        active_trace_context_id: int | None = None,
        active_trace_target_plans: list[list[int]] | None = None,
    ) -> tuple[Any, TopKReport]: ...

    def exact_top_k(
        self,
        *,
        steps: Any,
        initial_locations: Any,
        logit_scale: float,
        update_plan_timings: bool,
        use_shadow_prices: bool,
        top_k: int = 10,
        max_states: int = 2_000_000,
        n_threads: int | None = None,
        skip_infeasible: bool = False,
    ) -> tuple[Any, ExactTopKReport]: ...

    def exact_distribution(
        self,
        *,
        steps: Any,
        initial_locations: Any,
        logit_scale: float,
        update_plan_timings: bool,
        use_shadow_prices: bool,
        max_assignments: int = 100_000,
    ) -> dict[str, Any]: ...
