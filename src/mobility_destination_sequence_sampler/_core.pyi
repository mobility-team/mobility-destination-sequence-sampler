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
