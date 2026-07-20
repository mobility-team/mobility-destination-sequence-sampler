from __future__ import annotations

from typing import Any


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
        frontier_width: int = 32,
        proposal_limit_per_source: int = 16,
        stitch_bias: int = 0,
        continuation_state_limit: int = 1,
        continuation_proposal_limit: int = 1,
        seam_refresh_per_prefix: int = 1,
        top_k: int = 10,
        n_threads: int | None = None,
        skip_infeasible: bool = False,
        collect_profile: bool = False,
    ) -> tuple[Any, dict[str, int | float]]: ...

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
    ) -> tuple[Any, dict[str, int | str]]: ...


class ExperimentalDestinationSampler:
    def __init__(self, *, od_costs: Any, destination_inputs: Any) -> None: ...

    def sample_ternary_reference(
        self,
        *,
        steps: Any,
        initial_locations: Any,
        logit_scale: float,
        update_plan_timings: bool,
        use_shadow_prices: bool,
        seed: int,
        n_draws: int = 1,
        max_assignments: int = 100_000,
        skip_infeasible: bool = False,
    ) -> Any: ...

    def sample_particles(
        self,
        *,
        steps: Any,
        initial_locations: Any,
        logit_scale: float,
        update_plan_timings: bool,
        use_shadow_prices: bool,
        seed: int,
        n_particles: int = 32,
        candidate_count: int = 16,
        max_retries: int = 2,
        n_draws: int = 1,
        n_threads: int | None = None,
        skip_infeasible: bool = False,
    ) -> tuple[Any, dict[str, int | float]]: ...

    def search_bidirectional_top_k(
        self,
        *,
        steps: Any,
        initial_locations: Any,
        logit_scale: float,
        update_plan_timings: bool,
        use_shadow_prices: bool,
        seed: int,
        beam_width: int = 32,
        candidate_count: int = 16,
        top_k: int = 1,
        n_threads: int | None = None,
        skip_infeasible: bool = False,
        profile: bool = False,
    ) -> tuple[Any, dict[str, int | float]]: ...

    def search_ternary_top_k(
        self,
        *,
        steps: Any,
        initial_locations: Any,
        logit_scale: float,
        update_plan_timings: bool,
        use_shadow_prices: bool,
        k: int = 10,
        max_states: int = 2_000_000,
        n_threads: int | None = None,
        skip_infeasible: bool = False,
    ) -> tuple[Any, dict[str, int | str]]: ...
