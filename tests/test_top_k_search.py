from __future__ import annotations

import inspect

import polars as pl
import pytest

from experiments.top_k_config import ACTIVE_TOP_K_DEFAULTS
from mobility_destination_sequence_sampler import DestinationPlanSearch

from conftest import build_toy_inputs


def reference_steps() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    steps, initial_locations, od_costs, destination_inputs = build_toy_inputs()
    return (
        steps.with_columns(
            arrival_time=pl.col("departure_time") + 0.5,
            arrival_time_rigidity=pl.lit(0.5),
            departure_time_rigidity=pl.lit(0.5),
        ),
        initial_locations,
        od_costs,
        destination_inputs,
    )


def test_shared_experiment_defaults_match_the_live_top_k_signature() -> None:
    parameters = inspect.signature(DestinationPlanSearch.top_k).parameters
    assert {
        name: parameters[name].default for name in ACTIVE_TOP_K_DEFAULTS
    } == ACTIVE_TOP_K_DEFAULTS
    assert "continuation_log_gap" not in parameters
    assert "heuristic_reserve_limit" not in parameters
    assert "surface_bins" not in parameters


def test_two_step_top_k_matches_the_exact_oracle() -> None:
    steps, initial_locations, od_costs, destination_inputs = reference_steps()
    search = DestinationPlanSearch(od_costs=od_costs, destination_inputs=destination_inputs)
    common = {
        "steps": steps,
        "initial_locations": initial_locations,
        "logit_scale": 1.0,
        "update_plan_timings": True,
        "use_shadow_prices": False,
        "top_k": 2,
    }
    bounded, report = search.top_k(
        **common,
        exploration_seed=13,
        frontier_width=1,
        proposal_limit_per_source=1,
    )
    exact, oracle_report = search.exact_top_k(**common, max_states=100)

    def plans(frame: pl.DataFrame) -> list[tuple[int, ...]]:
        return [
            tuple(plan.sort("layer")["destination"].to_list())
            for plan in frame.partition_by("draw_id", maintain_order=True)
        ]

    assert plans(bounded) == plans(exact)
    assert oracle_report["incumbent_plans_seeded"] == 0
    assert report["complete_plan_candidates"] == 2
    assert report["forward_proposals_evaluated"] == 2
    assert report["stitch_pairs"] == 0
    assert report["active_trace_targets"] == []


def test_exact_oracle_rejects_negative_intermediate_duration() -> None:
    od_costs = pl.DataFrame(
        {
            "origin": [0, 0, 1, 2, 1],
            "destination": [1, 2, 1, 1, 0],
            "cost": [1.0, 0.0, 0.0, 0.0, 0.0],
            "time": [0.1, 10.0, 0.1, 10.0, 0.1],
        }
    )
    destination_inputs = pl.DataFrame(
        {
            "activity_id": [10, 10],
            "destination": [1, 2],
            "opportunity_capacity": [1.0, 1_000_000.0],
            "country_value_coefficient": [1.0, 1.0],
            "saturation_utility": [1.0, 1.0],
            "shadow_price": [0.0, 0.0],
        }
    )
    steps = pl.DataFrame(
        {
            "context_id": [1, 1, 1],
            "layer": [0, 1, 2],
            "activity_id": [10, 20, 0],
            "anchor_id": [None, None, None],
            "fixed_destination": [None, 1, 0],
            "departure_time": [8.0, 10.0, 17.0],
            "next_departure_time": [10.0, 17.0, 18.0],
            "duration_per_person": [2.0, 7.0, 1.0],
            "value_of_time": [1.0, 0.0, 0.0],
            "mean_duration_per_person": [1.0, 1.0, 1.0],
            "min_activity_time": [0.5, 0.5, 0.5],
            "arrival_time": [8.5, 10.5, 17.5],
            "arrival_time_rigidity": [0.5, 0.5, 0.0],
            "departure_time_rigidity": [0.5, 0.5, 0.5],
        }
    )
    common = {
        "steps": steps,
        "initial_locations": pl.DataFrame({"context_id": [1], "initial_zone": [0]}),
        "logit_scale": 1.0,
        "update_plan_timings": True,
        "use_shadow_prices": False,
        "top_k": 1,
    }
    search = DestinationPlanSearch(od_costs=od_costs, destination_inputs=destination_inputs)
    bounded, _ = search.top_k(
        **common,
        exploration_seed=13,
        frontier_width=8,
        proposal_limit_per_source=2,
    )
    exact, _ = search.exact_top_k(**common, max_states=100)

    expected = [1, 1, 0]
    assert bounded.sort(["draw_id", "layer"])["destination"].to_list() == expected
    assert exact.sort(["draw_id", "layer"])["destination"].to_list() == expected


def test_bidirectional_top_k_stitches_complete_plan() -> None:
    zones = [0, 1, 2]
    od_pairs = [
        (origin, destination)
        for origin in zones
        for destination in zones
        if (origin, destination) != (0, 2)
    ]
    od_costs = pl.DataFrame(
        {
            "origin": [origin for origin, _ in od_pairs],
            "destination": [destination for _, destination in od_pairs],
            "cost": [0.2 * abs(origin - destination) for origin, destination in od_pairs],
            "time": [0.1 * abs(origin - destination) for origin, destination in od_pairs],
        }
    )
    destination_inputs = pl.DataFrame(
        {
            "activity_id": [10, 10, 20, 20, 30, 30],
            "destination": [1, 2, 1, 2, 1, 2],
            "opportunity_capacity": [2.0, 1.0, 1.0, 2.0, 1.0, 1.0],
            "country_value_coefficient": [1.0] * 6,
            "saturation_utility": [1.0] * 6,
            "shadow_price": [0.0] * 6,
        }
    )
    steps = pl.DataFrame(
        {
            "context_id": [1] * 4,
            "layer": list(range(4)),
            "activity_id": [10, 20, 10, 0],
            "anchor_id": [99, None, 99, None],
            "fixed_destination": [None, None, None, 0],
            "departure_time": [8.0, 10.0, 12.0, 15.0],
            "arrival_time": [8.5, 10.5, 12.5, 15.5],
            "arrival_time_rigidity": [0.5, 0.5, 0.5, 0.0],
            "departure_time_rigidity": [0.5, 0.5, 0.5, 0.5],
            "next_departure_time": [10.0, 12.0, 15.0, 16.0],
            "duration_per_person": [2.0, 2.0, 3.0, 0.0],
            "value_of_time": [1.0, 1.0, 1.0, 0.0],
            "mean_duration_per_person": [1.0, 1.0, 1.0, 1.0],
            "min_activity_time": [0.5] * 4,
        }
    )
    search = DestinationPlanSearch(od_costs=od_costs, destination_inputs=destination_inputs)

    result, report = search.top_k(
        steps=steps,
        initial_locations=pl.DataFrame({"context_id": [1], "initial_zone": [0]}),
        logit_scale=1.0,
        update_plan_timings=True,
        use_shadow_prices=False,
        exploration_seed=13,
        frontier_width=8,
        proposal_limit_per_source=2,
        top_k=9,
    )

    assert result.group_by("draw_id").len()["len"].to_list() == [4] * min(
        report["complete_plan_candidates"], 9
    )
    assert 0 < report["stitch_pairs"] <= 64
    assert report["complete_plan_candidates"] >= 2
    assert report["continuation_proposals"] > 0
    assert report["seam_refresh_states"] >= 0
    symmetric_result, symmetric_report = search.top_k(
        steps=steps,
        initial_locations=pl.DataFrame({"context_id": [1], "initial_zone": [0]}),
        logit_scale=1.0,
        update_plan_timings=True,
        use_shadow_prices=False,
        exploration_seed=13,
        frontier_width=8,
        proposal_limit_per_source=2,
        candidate_strategy="symmetric_factor_map",
        top_k=9,
    )
    assert result.equals(symmetric_result)
    assert report["reverse_prefix_partial_calls"] == symmetric_report[
        "reverse_prefix_partial_calls"
    ]
    total_scores = (
        result.filter(pl.col("layer") == 0)
        .sort("draw_id")["total_log_weight"]
        .to_list()
    )
    assert total_scores == sorted(total_scores, reverse=True)
    for draw_id in result["draw_id"].unique().to_list():
        destinations = (
            result.filter(pl.col("draw_id") == draw_id)
            .sort("layer")["destination"]
            .to_list()
        )
        assert destinations[0] == destinations[2]

    terminal_result, _ = search.top_k(
        steps=steps.with_columns(
            fixed_destination=pl.when(pl.col("layer") == 3)
            .then(pl.lit(1))
            .otherwise(pl.col("fixed_destination"))
        ),
        initial_locations=pl.DataFrame({"context_id": [1], "initial_zone": [0]}),
        logit_scale=1.0,
        update_plan_timings=True,
        use_shadow_prices=False,
        exploration_seed=13,
        frontier_width=8,
        proposal_limit_per_source=2,
        top_k=9,
    )
    assert terminal_result.filter(pl.col("layer") == 3)["destination"].unique().to_list() == [1]

    isolated_steps = steps.with_columns(
        activity_id=pl.Series([10, 0, 20, 0]),
        anchor_id=pl.lit(None, dtype=pl.UInt32),
        fixed_destination=pl.Series([None, 0, None, 0], dtype=pl.UInt32),
    )
    routed_result, routed_report = search.top_k(
        steps=isolated_steps,
        initial_locations=pl.DataFrame({"context_id": [1], "initial_zone": [0]}),
        logit_scale=1.0,
        update_plan_timings=True,
        use_shadow_prices=False,
        exploration_seed=13,
        frontier_width=8,
        proposal_limit_per_source=2,
        candidate_strategy="adaptive_factor_map",
        top_k=9,
    )
    factor_result, factor_report = search.top_k(
        steps=isolated_steps,
        initial_locations=pl.DataFrame({"context_id": [1], "initial_zone": [0]}),
        logit_scale=1.0,
        update_plan_timings=True,
        use_shadow_prices=False,
        exploration_seed=13,
        frontier_width=8,
        proposal_limit_per_source=2,
        candidate_strategy="factor_map",
        top_k=9,
    )
    assert routed_result.equals(factor_result)
    for counter in (
        "forward_proposals_evaluated",
        "backward_proposals_evaluated",
        "factor_map_destinations_evaluated",
        "reverse_prefix_partial_calls",
    ):
        assert routed_report[counter] == factor_report[counter]

    missing_activity_search = DestinationPlanSearch(
        od_costs=od_costs,
        destination_inputs=destination_inputs.filter(pl.col("activity_id") != 20),
    )
    with pytest.raises(ValueError, match="no feasible destination sequence"):
        missing_activity_search.top_k(
            steps=steps,
            initial_locations=pl.DataFrame({"context_id": [1], "initial_zone": [0]}),
            logit_scale=1.0,
            update_plan_timings=True,
            use_shadow_prices=False,
            exploration_seed=13,
            frontier_width=8,
            proposal_limit_per_source=2,
            top_k=9,
        )


def test_bidirectional_top_k_supports_variable_anchor() -> None:
    steps, initial_locations, od_costs, destination_inputs = reference_steps()
    steps = pl.concat([steps, steps.tail(1)]).with_columns(
        layer=pl.int_range(pl.len(), dtype=pl.UInt32),
        context_id=pl.lit(1),
        activity_id=pl.when(pl.col("layer") == 0).then(10).otherwise(0),
        anchor_id=pl.when(pl.col("layer") == 0).then(99).otherwise(pl.lit(None, dtype=pl.UInt32)),
        fixed_destination=pl.when(pl.col("layer") == 0).then(pl.lit(None, dtype=pl.UInt32)).otherwise(0),
        arrival_time=pl.col("departure_time") + 0.5,
        arrival_time_rigidity=pl.lit(0.5),
        departure_time_rigidity=pl.lit(0.5),
    )
    search = DestinationPlanSearch(od_costs=od_costs, destination_inputs=destination_inputs)

    result, report = search.top_k(
        steps=steps,
        initial_locations=initial_locations.with_columns(
            context_id=pl.lit(1, dtype=pl.UInt64)
        ),
        logit_scale=1.0,
        update_plan_timings=True,
        use_shadow_prices=False,
        exploration_seed=13,
        skip_infeasible=True,
    )
    assert report["contexts"] == 1
    assert result.height in {0, steps.height}


def test_bidirectional_top_k_matches_exact_when_the_beam_covers_the_toy_domain() -> None:
    zones = [0, 1, 2]
    od_pairs = [(origin, destination) for origin in zones for destination in zones]
    od_costs = pl.DataFrame(
        {
            "origin": [origin for origin, _ in od_pairs],
            "destination": [destination for _, destination in od_pairs],
            "cost": [0.2 * abs(origin - destination) for origin, destination in od_pairs],
            "time": [0.1 * abs(origin - destination) for origin, destination in od_pairs],
        }
    )
    destination_inputs = pl.DataFrame(
        {
            "activity_id": [10, 10, 20, 20],
            "destination": [1, 2, 1, 2],
            "opportunity_capacity": [2.0, 1.0, 1.0, 2.0],
            "country_value_coefficient": [1.0] * 4,
            "saturation_utility": [1.0] * 4,
            "shadow_price": [0.0] * 4,
        }
    )
    steps = pl.DataFrame(
        {
            "context_id": [1] * 4,
            "layer": list(range(4)),
            "activity_id": [10, 20, 10, 0],
            "anchor_id": [None] * 4,
            "fixed_destination": [None, None, None, 0],
            "departure_time": [8.0, 10.0, 12.0, 15.0],
            "arrival_time": [8.5, 10.5, 12.5, 15.5],
            "arrival_time_rigidity": [0.5, 0.5, 0.5, 0.0],
            "departure_time_rigidity": [0.5, 0.5, 0.5, 0.5],
            "next_departure_time": [10.0, 12.0, 15.0, 16.0],
            "duration_per_person": [2.0, 2.0, 3.0, 0.0],
            "value_of_time": [1.0, 1.0, 1.0, 0.0],
            "mean_duration_per_person": [1.0, 1.0, 1.0, 1.0],
            "min_activity_time": [0.5] * 4,
        }
    )
    initial_locations = pl.DataFrame({"context_id": [1], "initial_zone": [0]})
    search = DestinationPlanSearch(od_costs=od_costs, destination_inputs=destination_inputs)
    common = {
        "steps": steps,
        "initial_locations": initial_locations,
        "logit_scale": 1.0,
        "update_plan_timings": True,
        "use_shadow_prices": False,
        "top_k": 8,
    }

    bounded, _ = search.top_k(
        **common,
        exploration_seed=13,
        frontier_width=8,
        proposal_limit_per_source=8,
        candidate_strategy="factor_map",
        seam_refresh_per_prefix=0,
    )
    priced, pricing_report = search.top_k(
        **common,
        exploration_seed=13,
        frontier_width=8,
        proposal_limit_per_source=8,
        candidate_strategy="factor_map",
        seam_refresh_per_prefix=0,
        pricing_passes=2,
        pricing_pair_candidate_limit=2,
        pricing_min_layers=4,
    )
    exact, seeded_report = search.exact_top_k(**common, max_states=100)
    exact_without_seed, unseeded_report = search.exact_top_k(
        **common,
        max_states=100,
        use_bounded_incumbent=False,
    )

    def plans(frame: pl.DataFrame) -> list[tuple[int, ...]]:
        return [
            tuple(plan.sort("layer")["destination"].to_list())
            for plan in frame.partition_by("draw_id", maintain_order=True)
        ]

    assert plans(bounded) == plans(priced) == plans(exact) == plans(exact_without_seed)
    assert pricing_report["pricing_candidate_evaluations"] > 0
    assert pricing_report["pricing_feasible_evaluations"] > 0
    assert pricing_report["pricing_pair_evaluations"] > 0
    assert pricing_report["pricing_pair_feasible_evaluations"] > 0
    assert pricing_report["pricing_rounds"] == 1
    assert seeded_report["incumbent_plans_seeded"] == 8
    assert unseeded_report["incumbent_plans_seeded"] == 0


def test_interacting_pair_pricing_crosses_a_single_replacement_valley() -> None:
    zones = [0, 1, 2]
    pair_cost = {
        (0, 0): 0.0,
        (0, 1): 0.0,
        (0, 2): 2.0,
        (1, 0): 0.0,
        (1, 1): 0.0,
        (1, 2): 10.0,
        (2, 0): 2.0,
        (2, 1): 10.0,
        (2, 2): 0.0,
    }
    od_pairs = [(origin, destination) for origin in zones for destination in zones]
    od_costs = pl.DataFrame(
        {
            "origin": [origin for origin, _ in od_pairs],
            "destination": [destination for _, destination in od_pairs],
            "cost": [pair_cost[pair] for pair in od_pairs],
            "time": [0.0] * len(od_pairs),
        }
    )
    destination_inputs = pl.DataFrame(
        {
            "activity_id": [10, 10, 20, 20],
            "destination": [1, 2, 1, 2],
            "opportunity_capacity": [100.0, 1.0, 100.0, 1.0],
            "country_value_coefficient": [0.0] * 4,
            "saturation_utility": [1.0] * 4,
            "shadow_price": [0.0, 5.0, 0.0, 5.0],
        }
    )
    steps = pl.DataFrame(
        {
            "context_id": [1] * 3,
            "layer": [0, 1, 2],
            "activity_id": [10, 20, 0],
            "anchor_id": [None] * 3,
            "fixed_destination": [None, None, 0],
            "departure_time": [8.0, 11.0, 14.0],
            "arrival_time": [8.0, 11.0, 14.0],
            "arrival_time_rigidity": [0.0] * 3,
            "departure_time_rigidity": [0.0] * 3,
            "next_departure_time": [11.0, 14.0, 15.0],
            "duration_per_person": [2.0, 2.0, 0.0],
            "value_of_time": [1.0, 1.0, 0.0],
            "mean_duration_per_person": [1.0] * 3,
            "min_activity_time": [0.5] * 3,
        }
    )
    common = {
        "steps": steps,
        "initial_locations": pl.DataFrame({"context_id": [1], "initial_zone": [0]}),
        "logit_scale": 1.0,
        "update_plan_timings": False,
        "use_shadow_prices": True,
        "top_k": 1,
    }
    search = DestinationPlanSearch(od_costs=od_costs, destination_inputs=destination_inputs)
    pricing_options = {
        "exploration_seed": 13,
        "frontier_width": 1,
        "proposal_limit_per_source": 1,
        "candidate_strategy": "heuristic",
        "continuation_state_limit": 1,
        "continuation_proposal_limit": 1,
        "seam_refresh_per_prefix": 0,
        "pricing_passes": 1,
        "pricing_seed_limit": 1,
        "pricing_column_limit": 1,
        "pricing_pair_deep_candidate_limit": 0,
        "pricing_min_layers": 3,
    }
    scalar, _ = search.top_k(
        **common,
        **pricing_options,
        pricing_pair_candidate_limit=0,
    )
    paired, report = search.top_k(
        **common,
        **pricing_options,
        pricing_pair_candidate_limit=1,
    )
    adaptive_options = {
        **pricing_options,
        "pricing_pair_candidate_limit": 1,
        "pricing_pair_deep_candidate_limit": 2,
        "pricing_pair_deep_min_layers": 0,
    }
    adaptive, adaptive_report = search.top_k(
        **common,
        **adaptive_options,
        active_trace_context_id=1,
        active_trace_target_plans=[],
    )
    exact, _ = search.exact_top_k(
        **common,
        max_states=100,
        use_bounded_incumbent=False,
    )

    def plan(frame: pl.DataFrame) -> tuple[int, ...]:
        return tuple(frame.sort("layer")["destination"].to_list())

    assert plan(scalar) == (1, 1, 0)
    assert plan(paired) == plan(adaptive) == plan(exact) == (2, 2, 0)
    assert report["pricing_pair_plans_added"] > 0
    assert adaptive_report["pricing_pair_probes"] > 0
    assert adaptive_report["pricing_pair_expansions"] > 0
    probe = adaptive_report["pricing_pair_probe_reports"][0]
    assert probe["evaluated"] > 0
    assert 0 < probe["feasible"] <= probe["evaluated"]
    assert probe["kth_score_improvement"] >= 0.0


def test_home_bounded_factor_maps_keep_a_cross_home_anchor() -> None:
    zones = [0, 1, 2]
    od_costs = pl.DataFrame(
        {
            "origin": [origin for origin in zones for _ in zones],
            "destination": [destination for _ in zones for destination in zones],
            "cost": [0.2 * abs(origin - destination) for origin in zones for destination in zones],
            "time": [0.1 * abs(origin - destination) for origin in zones for destination in zones],
        }
    )
    destination_inputs = pl.DataFrame(
        {
            "activity_id": [10, 10],
            "destination": [1, 2],
            "opportunity_capacity": [2.0, 1.0],
            "country_value_coefficient": [1.0, 1.0],
            "saturation_utility": [1.0, 1.0],
            "shadow_price": [0.0, 0.0],
        }
    )
    steps = pl.DataFrame(
        {
            "context_id": [1] * 6,
            "layer": list(range(6)),
            "activity_id": [10, 10, 0, 10, 10, 0],
            "anchor_id": [91, None, None, 91, None, None],
            "fixed_destination": [None, None, 0, None, None, 0],
            "departure_time": [8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
            "arrival_time": [8.5, 10.5, 12.5, 14.5, 16.5, 18.5],
            "arrival_time_rigidity": [0.5, 0.5, 0.0, 0.5, 0.5, 0.0],
            "departure_time_rigidity": [0.5] * 6,
            "next_departure_time": [10.0, 12.0, 14.0, 16.0, 18.0, 19.0],
            "duration_per_person": [2.0] * 5 + [0.0],
            "value_of_time": [1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            "mean_duration_per_person": [1.0] * 6,
            "min_activity_time": [0.5] * 6,
        }
    )
    initial_locations = pl.DataFrame({"context_id": [1], "initial_zone": [0]})
    search = DestinationPlanSearch(od_costs=od_costs, destination_inputs=destination_inputs)
    common = {
        "steps": steps,
        "initial_locations": initial_locations,
        "logit_scale": 1.0,
        "update_plan_timings": True,
        "use_shadow_prices": False,
        "top_k": 8,
    }

    bounded, report = search.top_k(
        **common,
        exploration_seed=13,
        frontier_width=8,
        proposal_limit_per_source=8,
        candidate_strategy="symmetric_factor_map",
        factor_map_max_depth=3,
        seam_refresh_per_prefix=0,
    )
    exact, _ = search.exact_top_k(**common, max_states=100)

    def plans(frame: pl.DataFrame) -> list[tuple[int, ...]]:
        return [
            tuple(plan.sort("layer")["destination"].to_list())
            for plan in frame.partition_by("draw_id", maintain_order=True)
        ]

    assert plans(bounded) == plans(exact)
    assert report["factor_map_destinations_evaluated"] > 0
    for plan in plans(bounded):
        assert plan[2] == plan[5] == 0
        assert plan[0] == plan[3]


def test_exact_oracle_fails_explicitly_at_its_state_budget() -> None:
    steps, initial_locations, od_costs, destination_inputs = reference_steps()
    search = DestinationPlanSearch(od_costs=od_costs, destination_inputs=destination_inputs)
    steps = pl.concat([steps.head(1), steps]).with_columns(
        layer=pl.int_range(pl.len(), dtype=pl.UInt32),
        activity_id=pl.when(pl.col("layer") == 2).then(0).otherwise(10),
        fixed_destination=pl.when(pl.col("layer") == 2)
        .then(0)
        .otherwise(pl.lit(None, dtype=pl.UInt32)),
        departure_time=pl.Series([8.0, 10.0, 17.0]),
        next_departure_time=pl.Series([10.0, 17.0, 18.0]),
        arrival_time=pl.Series([8.5, 10.5, 17.5]),
    )

    with pytest.raises(ValueError, match="exceeded max_states=1"):
        search.exact_top_k(
            steps=steps,
            initial_locations=initial_locations,
            logit_scale=1.0,
            update_plan_timings=True,
            use_shadow_prices=False,
            top_k=1,
            max_states=1,
            use_bounded_incumbent=False,
        )


def test_search_requires_both_timing_rigidities() -> None:
    steps, initial_locations, od_costs, destination_inputs = reference_steps()
    search = DestinationPlanSearch(od_costs=od_costs, destination_inputs=destination_inputs)

    with pytest.raises(ValueError, match="departure_time_rigidity"):
        search.top_k(
            steps=steps.drop("departure_time_rigidity"),
            initial_locations=initial_locations,
            logit_scale=1.0,
            update_plan_timings=True,
            use_shadow_prices=False,
            exploration_seed=13,
        )
