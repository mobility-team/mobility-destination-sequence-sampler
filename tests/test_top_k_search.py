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
    exact, _ = search.exact_top_k(**common, max_states=100)

    def plans(frame: pl.DataFrame) -> list[tuple[int, ...]]:
        return [
            tuple(plan.sort("layer")["destination"].to_list())
            for plan in frame.partition_by("draw_id", maintain_order=True)
        ]

    assert plans(bounded) == plans(exact)
    assert report["complete_plan_candidates"] == 2
    assert report["forward_proposals_evaluated"] == 2
    assert report["stitch_pairs"] == 0
    assert report["heuristic_reserve_triggers"] == 0
    assert report["heuristic_reserve_proposals"] == 0
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
    exact, _ = search.exact_top_k(**common, max_states=100)

    def plans(frame: pl.DataFrame) -> list[tuple[int, ...]]:
        return [
            tuple(plan.sort("layer")["destination"].to_list())
            for plan in frame.partition_by("draw_id", maintain_order=True)
        ]

    assert plans(bounded) == plans(exact)


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
