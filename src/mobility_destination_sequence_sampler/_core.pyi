from __future__ import annotations

from typing import Any, Literal, TypedDict


CandidateStrategy = Literal[
    "heuristic",
    "factor_map",
    "symmetric_factor_map",
    "adaptive_factor_map",
]


class ActiveTraceTargetReport(TypedDict):
    zones: list[int]
    proposed: list[bool]
    retained: list[bool]
    prefix_proposed: list[bool | None]
    prefix_retained: list[bool | None]
    guidance_retained: list[bool]
    guidance_proposed: list[bool]
    exact_guidance_rank: list[int | None]
    exact_guidance_log_gap: list[float | None]
    prefix_pruned: list[bool | None]
    pruned: list[bool]


class PricingPairProbeReport(TypedDict):
    pass_index: int
    seed_rank: int
    left_group: int
    right_group: int
    evaluated: int
    feasible: int
    expansion_evaluated: int
    boundary_score_gap: float | None
    neighborhood_saturated: bool
    entering_working_top_k: int
    kth_score_improvement: float
    max_non_additivity: float
    expansion_entering_working_top_k: int
    expansion_kth_score_improvement: float


class TopKReport(TypedDict):
    contexts: int
    forward_proposals_evaluated: int
    backward_proposals_evaluated: int
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
    pricing_candidate_evaluations: int
    pricing_feasible_evaluations: int
    pricing_plans_added: int
    pricing_pair_evaluations: int
    pricing_pair_feasible_evaluations: int
    pricing_pair_plans_added: int
    pricing_pair_probes: int
    pricing_pair_expansions: int
    pricing_pair_probe_reports: list[PricingPairProbeReport]
    pricing_rounds: int
    stitch_pairs: int
    complete_plan_candidates: int
    infeasible_contexts: int
    build_problem_ns: int
    backward_search_ns: int
    backward_guidance_ns: int
    forward_search_ns: int
    continuation_guidance_ns: int
    factor_map_ns: int
    seam_refresh_ns: int
    pricing_ns: int
    stitch_ns: int
    materialize_ns: int
    total_search_ns: int
    active_trace_targets: list[ActiveTraceTargetReport]


class ExactTopKReport(TypedDict):
    contexts: int
    split_contexts: int
    conditioned_anchor_contexts: int
    anchor_conditions_considered: int
    anchor_conditions_pruned: int
    incumbent_contexts: int
    incumbent_plans_seeded: int
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
        symmetric_forward_proposal_limit: int = 20,
        candidate_strategy: CandidateStrategy = "adaptive_factor_map",
        factor_map_max_depth: int = 5,
        stitch_bias: int = 1,
        continuation_state_limit: int = 1,
        deep_continuation_state_limit: int = 2,
        continuation_proposal_limit: int = 1,
        seam_refresh_per_prefix: int = 1,
        pricing_passes: int = 2,
        pricing_seed_limit: int = 10,
        pricing_column_limit: int = 4,
        pricing_pair_candidate_limit: int = 4,
        pricing_pair_deep_candidate_limit: int = 8,
        pricing_pair_deep_min_layers: int = 0,
        pricing_next_pass_min_new: int = 3,
        pricing_min_layers: int = 6,
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
        use_bounded_incumbent: bool = True,
    ) -> tuple[Any, ExactTopKReport]: ...
