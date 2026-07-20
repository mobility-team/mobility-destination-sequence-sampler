from __future__ import annotations

import math
from collections import Counter
from itertools import product

import polars as pl
import pytest

from mobility_destination_sequence_sampler._core import (
    ExperimentalDestinationSampler as DestinationSampler,
    sample_destination_sequences,
)

from conftest import build_toy_inputs, rows_by_draw, toy_path_log_weights


def test_sampled_path_weights_match_python_reference() -> None:
    steps, initial_locations, od_costs, destination_inputs = build_toy_inputs()
    result = sample_destination_sequences(
        steps=steps,
        initial_locations=initial_locations,
        od_costs=od_costs,
        destination_inputs=destination_inputs,
        logit_scale=1.0,
        update_plan_timings=True,
        use_shadow_prices=False,
        seed=42,
        n_draws=20,
    )
    expected = toy_path_log_weights()

    assert result.height == 40
    for draw in rows_by_draw(result):
        destination = draw["destination"][0]
        sampled_path_weight = draw["local_log_weight"].sum()
        assert math.isclose(
            sampled_path_weight,
            expected[destination],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        assert draw["destination"][-1] == 0


def test_profiled_sample_matches_normal_sample() -> None:
    steps, initial_locations, od_costs, destination_inputs = build_toy_inputs()
    sampler = DestinationSampler(
        od_costs=od_costs,
        destination_inputs=destination_inputs,
    )
    arguments = {
        "steps": steps,
        "initial_locations": initial_locations,
        "logit_scale": 1.0,
        "update_plan_timings": True,
        "use_shadow_prices": False,
        "seed": 42,
        "n_draws": 5,
    }

    expected = sampler.sample(**arguments)
    result, profile = sampler.sample_with_profile(**arguments)

    assert result.equals(expected)
    assert profile["contexts"] == 1
    assert profile["successful_contexts"] == 1
    assert profile["output_rows"] == result.height
    assert profile["sampling_wall_seconds"] >= 0.0


def test_ternary_reference_applies_both_adjacent_rigidity_adjustments() -> None:
    od_costs = pl.DataFrame(
        {
            "origin": [0, 1, 2],
            "destination": [1, 2, 0],
            "cost": [0.0, 0.0, 0.0],
            "time": [2.0, 1.5, 0.5],
        }
    )
    destination_inputs = pl.DataFrame(
        {
            "activity_id": [10, 20],
            "destination": [1, 2],
            "opportunity_capacity": [1.0, 1.0],
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
            "fixed_destination": [None, None, 0],
            "departure_time": [8.0, 17.0, 19.0],
            "arrival_time": [9.0, 17.5, 19.5],
            "arrival_time_rigidity": [0.5, 0.25, 1.0],
            "next_departure_time": [17.0, 19.0, 20.0],
            "duration_per_person": [8.0, 1.5, 0.5],
            "value_of_time": [1.0, 1.0, 0.0],
            "mean_duration_per_person": [1.0, 1.0, 1.0],
            "min_activity_time": [1.0, 1.0, 1.0],
        }
    )
    initial_locations = pl.DataFrame(
        {"context_id": [1], "initial_zone": [0]}
    )
    sampler = DestinationSampler(
        od_costs=od_costs,
        destination_inputs=destination_inputs,
    )

    result = sampler.sample_ternary_reference(
        steps=steps,
        initial_locations=initial_locations,
        logit_scale=1.0,
        update_plan_timings=True,
        use_shadow_prices=False,
        seed=1,
    ).sort("layer")

    # Activity 10 loses half of the extra incoming hour and one quarter of
    # the extra outgoing hour: 8 - 0.5 - 0.25 = 7.25 hours.
    assert result["local_log_weight"][0] == pytest.approx(math.log(7.25))
    # Activity 20 loses 75% of its extra incoming hour; the return-home trip
    # matches its reference time: 1.5 - 0.75 = 0.75 hours.
    assert result["local_log_weight"][1] == pytest.approx(0.0)


def archived_second_order_solver_matches_repeated_anchor_enumeration() -> None:
    zones = [0, 1, 2]
    costs = {
        (origin, destination): 0.2 + abs(origin - destination) * 0.3
        for origin in zones
        for destination in zones
    }
    times = {
        (origin, destination): 0.1 + abs(origin - destination) * 0.4
        for origin in zones
        for destination in zones
    }
    od_costs = pl.DataFrame(
        {
            "origin": [origin for origin in zones for _ in zones],
            "destination": [
                destination for _ in zones for destination in zones
            ],
            "cost": list(costs.values()),
            "time": list(times.values()),
        }
    )
    destination_inputs = pl.DataFrame(
        {
            "activity_id": [10, 10, 20, 20],
            "destination": [1, 2, 1, 2],
            "opportunity_capacity": [2.0, 1.0, 1.0, 3.0],
            "country_value_coefficient": [1.0] * 4,
            "saturation_utility": [1.0] * 4,
            "shadow_price": [0.0] * 4,
        }
    )
    steps = pl.DataFrame(
        {
            "context_id": [7] * 4,
            "layer": [0, 1, 2, 3],
            "activity_id": [10, 20, 10, 0],
            "anchor_id": [10, None, 10, None],
            "fixed_destination": [None, None, None, 0],
            "departure_time": [8.0, 10.0, 12.0, 14.0],
            "arrival_time": [8.5, 10.5, 12.5, 14.5],
            "arrival_time_rigidity": [0.5, 0.25, 0.75, 1.0],
            "next_departure_time": [10.0, 12.0, 14.0, 15.0],
            "duration_per_person": [1.5, 1.5, 1.5, 0.5],
            "value_of_time": [1.0, 1.0, 1.0, 0.0],
            "mean_duration_per_person": [1.0] * 4,
            "min_activity_time": [0.5] * 4,
        }
    )
    sampler = DestinationSampler(
        od_costs=od_costs,
        destination_inputs=destination_inputs,
    )
    result = sampler.solve_second_order(
        steps=steps,
        initial_locations=pl.DataFrame(
            {"context_id": [7], "initial_zone": [0]}
        ),
        logit_scale=1.0,
        update_plan_timings=True,
        use_shadow_prices=False,
    )

    capacities = {1: 2.0, 2: 1.0}
    middle_capacities = {1: 1.0, 2: 3.0}
    scores: list[tuple[int, float]] = []
    for anchor, middle in product([1, 2], repeat=2):
        path = [anchor, middle, anchor, 0]
        origins = [0, *path[:-1]]
        adjusted_departures = []
        adjusted_arrivals = []
        for layer, (origin, destination) in enumerate(
            zip(origins, path, strict=True)
        ):
            reference_time = 0.5
            delta = times[origin, destination] - reference_time
            rigidity = steps["arrival_time_rigidity"][layer]
            adjusted_departures.append(
                steps["departure_time"][layer] - rigidity * delta
            )
            adjusted_arrivals.append(
                steps["arrival_time"][layer]
                + (1.0 - rigidity) * delta
            )
        score = math.log(capacities[anchor]) + math.log(
            middle_capacities[middle]
        )
        for layer, (origin, destination) in enumerate(
            zip(origins, path, strict=True)
        ):
            duration = max(
                (
                    adjusted_departures[layer + 1]
                    if layer + 1 < len(path)
                    else steps["next_departure_time"][layer]
                )
                - adjusted_arrivals[layer],
                1e-3,
            )
            activity_utility = (
                steps["value_of_time"][layer]
                * max(
                    math.log(
                        duration / steps["min_activity_time"][layer]
                    ),
                    0.0,
                )
            )
            score += activity_utility - costs[origin, destination]
        scores.append((anchor, score))

    maximum = max(score for _, score in scores)
    weights = [math.exp(score - maximum) for _, score in scores]
    log_partition = maximum + math.log(sum(weights))
    first_probability = {
        anchor: sum(
            weight
            for (candidate, _), weight in zip(
                scores, weights, strict=True
            )
            if candidate == anchor
        )
        / sum(weights)
        for anchor in [1, 2]
    }

    assert result["log_partitions"][0] == pytest.approx(log_partition)
    probabilities = result["first_destination_probabilities"]
    assert probabilities[0] == 0.0
    assert probabilities[1] == pytest.approx(first_probability[1])
    assert probabilities[2] == pytest.approx(first_probability[2])

    without_forward_feasibility = sampler.solve_second_order(
        steps=steps,
        initial_locations=pl.DataFrame(
            {"context_id": [7], "initial_zone": [0]}
        ),
        logit_scale=1.0,
        update_plan_timings=True,
        use_shadow_prices=False,
        use_bidirectional_feasibility=False,
    )
    assert without_forward_feasibility["log_partitions"] == pytest.approx(
        result["log_partitions"]
    )
    assert without_forward_feasibility["first_destination_probabilities"] == pytest.approx(
        result["first_destination_probabilities"]
    )
    assert result["forward_pair_states"] > 0
    assert without_forward_feasibility["forward_pair_states"] == 0


def archived_second_order_solver_rejects_a_locally_infeasible_destination() -> None:
    od_costs = pl.DataFrame(
        {
            "origin": [0, 0],
            "destination": [1, 2],
            "cost": [0.0, 0.0],
            "time": [0.25, 0.45],
        }
    )
    destination_inputs = pl.DataFrame(
        {
            "activity_id": [10, 10],
            "destination": [1, 2],
            "opportunity_capacity": [1.0, 1.0],
            "country_value_coefficient": [1.0, 1.0],
            "saturation_utility": [1.0, 1.0],
            "shadow_price": [0.0, 0.0],
        }
    )
    steps = pl.DataFrame(
        {
            "context_id": [1],
            "layer": [0],
            "activity_id": [10],
            "anchor_id": [None],
            "fixed_destination": [None],
            "departure_time": [8.0],
            "arrival_time": [8.25],
            "arrival_time_rigidity": [0.0],
            "next_departure_time": [8.3],
            "duration_per_person": [0.05],
            "value_of_time": [1.0],
            "mean_duration_per_person": [1.0],
            "min_activity_time": [0.1],
        }
    )
    sampler = DestinationSampler(
        od_costs=od_costs,
        destination_inputs=destination_inputs,
    )

    result = sampler.solve_second_order(
        steps=steps,
        initial_locations=pl.DataFrame(
            {"context_id": [1], "initial_zone": [0]}
        ),
        logit_scale=1.0,
        update_plan_timings=True,
        use_shadow_prices=False,
    )

    assert result["first_destination_probabilities"] == [0.0, 1.0, 0.0]
    assert result["duration_checks"] == 2
    assert result["duration_infeasible"] == 1
    assert result["pair_states"] == 2
    assert result["feasible_pair_states"] == 1


def archived_second_order_solver_applies_wrapped_home_shadow_price_at_day_boundaries() -> None:
    """Longer morning and evening trips lose wrapped home time cheaply."""
    od_costs = pl.DataFrame(
        {
            "origin": [0, 0, 1, 2],
            "destination": [1, 2, 0, 0],
            "cost": [0.0, 0.0, 0.0, 0.0],
            "time": [1.0, 2.0, 1.0, 2.0],
        }
    )
    destination_inputs = pl.DataFrame(
        {
            "activity_id": [10, 10],
            "destination": [1, 2],
            "opportunity_capacity": [1.0, 1.0],
            "country_value_coefficient": [1.0, 1.0],
            "saturation_utility": [1.0, 1.0],
            "shadow_price": [0.0, 0.0],
        }
    )
    steps = pl.DataFrame(
        {
            "context_id": [1, 1],
            "layer": [0, 1],
            "activity_id": [10, 0],
            "anchor_id": [None, None],
            "fixed_destination": [None, 0],
            "departure_time": [8.0, 17.0],
            "arrival_time": [9.0, 18.0],
            "arrival_time_rigidity": [1.0, 0.0],
            "departure_time_rigidity": [0.0, 1.0],
            "next_departure_time": [17.0, 18.0],
            "duration_per_person": [8.0, 0.0],
            "value_of_time": [0.0, 0.0],
            "mean_duration_per_person": [1.0, 1.0],
            "min_activity_time": [1.0, 1.0],
        }
    )
    sampler = DestinationSampler(
        od_costs=od_costs,
        destination_inputs=destination_inputs,
    )
    arguments = {
        "steps": steps,
        "initial_locations": pl.DataFrame(
            {"context_id": [1], "initial_zone": [0]}
        ),
        "logit_scale": 1.0,
        "update_plan_timings": True,
        "use_shadow_prices": False,
    }

    without_shadow = sampler.solve_second_order(**arguments)
    with_shadow = sampler.solve_second_order(
        **arguments,
        wrapped_home_time_shadow_price=2.0,
    )

    assert without_shadow["first_destination_probabilities"][1:] == pytest.approx(
        [0.5, 0.5]
    )
    assert with_shadow["first_destination_probabilities"][1] == pytest.approx(
        1.0 / (1.0 + math.exp(-4.0))
    )
    assert with_shadow["first_destination_probabilities"][2] == pytest.approx(
        1.0 / (1.0 + math.exp(4.0))
    )


def test_ternary_reference_rejects_large_assignment_lattices() -> None:
    steps, initial_locations, od_costs, destination_inputs = build_toy_inputs()
    steps = steps.with_columns(
        arrival_time=pl.col("departure_time") + pl.Series([0.25, 0.25]),
        arrival_time_rigidity=pl.lit(0.0),
    )
    sampler = DestinationSampler(
        od_costs=od_costs,
        destination_inputs=destination_inputs,
    )

    with pytest.raises(ValueError, match="more than 1 complete destination assignments"):
        sampler.sample_ternary_reference(
            steps=steps,
            initial_locations=initial_locations,
            logit_scale=1.0,
            update_plan_timings=True,
            use_shadow_prices=False,
            seed=1,
            max_assignments=1,
        )


def test_ternary_reference_splits_independent_home_tours() -> None:
    zones = [0, 1, 2]
    od_costs = pl.DataFrame(
        {
            "origin": [origin for origin in zones for _ in zones],
            "destination": [destination for _ in zones for destination in zones],
            "cost": [
                abs(destination - origin) * 0.1
                for origin in zones
                for destination in zones
            ],
            "time": [
                abs(destination - origin) * 0.1
                for origin in zones
                for destination in zones
            ],
        }
    )
    destination_inputs = pl.DataFrame(
        {
            "activity_id": [10, 10, 20, 20],
            "destination": [1, 2, 1, 2],
            "opportunity_capacity": [1.0, 1.0, 1.0, 1.0],
            "country_value_coefficient": [1.0] * 4,
            "saturation_utility": [1.0] * 4,
            "shadow_price": [0.0] * 4,
        }
    )
    steps = pl.DataFrame(
        {
            "context_id": [1] * 6,
            "layer": list(range(6)),
            "activity_id": [10, 20, 0, 10, 20, 0],
            "anchor_id": [None] * 6,
            "fixed_destination": [None, None, 0, None, None, 0],
            "departure_time": [8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
            "arrival_time": [8.5, 10.5, 12.5, 14.5, 16.5, 18.5],
            "arrival_time_rigidity": [0.5] * 6,
            "next_departure_time": [10.0, 12.0, 14.0, 16.0, 18.0, 19.0],
            "duration_per_person": [1.5] * 6,
            "value_of_time": [1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            "mean_duration_per_person": [1.0] * 6,
            "min_activity_time": [0.5] * 6,
        }
    )
    sampler = DestinationSampler(
        od_costs=od_costs,
        destination_inputs=destination_inputs,
    )

    result = sampler.sample_ternary_reference(
        steps=steps,
        initial_locations=pl.DataFrame(
            {"context_id": [1], "initial_zone": [0]}
        ),
        logit_scale=1.0,
        update_plan_timings=True,
        use_shadow_prices=False,
        seed=4,
        n_draws=10_000,
        max_assignments=8,
    )

    # The unsplit lattice has 2^4 = 16 assignments. Splitting at the
    # intermediate home enumerates two 2^2 tours, exactly eight configurations.
    assert result.height == 60_000
    first_draw = result.filter(pl.col("draw_id") == 1).sort("layer")
    assert first_draw["destination"].gather([2, 5]).to_list() == [0, 0]

    sampled_plans = (
        result.filter(pl.col("layer").is_in([0, 1, 3, 4]))
        .sort(["draw_id", "layer"])
        .group_by("draw_id", maintain_order=True)
        .agg(pl.col("destination"))
    )
    plan_weights = result.filter(pl.col("layer") == 0).sort("draw_id")[
        "total_log_weight"
    ]
    counts = Counter(tuple(plan) for plan in sampled_plans["destination"])
    weights = {}
    for plan, weight in zip(sampled_plans["destination"], plan_weights, strict=True):
        weights[tuple(plan)] = weight
    maximum = max(weights.values())
    normalizer = sum(math.exp(weight - maximum) for weight in weights.values())
    for plan, weight in weights.items():
        observed = counts[plan] / 10_000
        expected = math.exp(weight - maximum) / normalizer
        assert observed == pytest.approx(expected, abs=0.02)

    heap_steps = steps.with_columns(
        arrival_time_rigidity=pl.when(pl.col("layer") == 3)
        .then(0.0)
        .otherwise(pl.col("arrival_time_rigidity"))
    )
    ranked, report = sampler.search_ternary_top_k(
        steps=heap_steps,
        initial_locations=pl.DataFrame(
            {"context_id": [1], "initial_zone": [0]}
        ),
        logit_scale=1.0,
        update_plan_timings=True,
        use_shadow_prices=False,
        k=16,
    )
    ranked_plans = (
        ranked.filter(pl.col("layer").is_in([0, 1, 3, 4]))
        .sort(["draw_id", "layer"])
        .group_by("draw_id", maintain_order=True)
        .agg(pl.col("destination"))
    )
    assert ranked_plans.height == 16
    assert len({tuple(plan) for plan in ranked_plans["destination"]}) == 16
    assert report["split_contexts"] == 1


def test_heap_reference_returns_exact_top_k_complete_plans() -> None:
    zones = [0, 1, 2]
    od_costs = pl.DataFrame(
        {
            "origin": [origin for origin in zones for _ in zones],
            "destination": [destination for _ in zones for destination in zones],
            "cost": [
                0.2 * abs(destination - origin)
                for origin in zones
                for destination in zones
            ],
            "time": [
                0.1 * abs(destination - origin)
                for origin in zones
                for destination in zones
            ],
        }
    )
    destination_inputs = pl.DataFrame(
        {
            "activity_id": [10, 10, 20, 20],
            "destination": [1, 2, 1, 2],
            "opportunity_capacity": [4.0, 1.0, 1.0, 3.0],
            "country_value_coefficient": [1.0] * 4,
            "saturation_utility": [1.0] * 4,
            "shadow_price": [0.0] * 4,
        }
    )
    steps = pl.DataFrame(
        {
            "context_id": [1, 1, 1],
            "layer": [0, 1, 2],
            "activity_id": [10, 20, 0],
            "anchor_id": [None, None, None],
            "fixed_destination": [None, None, 0],
            "departure_time": [8.0, 12.0, 17.0],
            "arrival_time": [8.5, 12.5, 17.5],
            "arrival_time_rigidity": [0.8, 0.2, 1.0],
            "next_departure_time": [12.0, 17.0, 18.0],
            "duration_per_person": [3.5, 4.5, 0.5],
            "value_of_time": [2.0, 3.0, 0.0],
            "mean_duration_per_person": [2.0, 3.0, 1.0],
            "min_activity_time": [0.5, 0.5, 1.0],
        }
    )
    sampler = DestinationSampler(
        od_costs=od_costs,
        destination_inputs=destination_inputs,
    )

    result, report = sampler.search_ternary_top_k(
        steps=steps,
        initial_locations=pl.DataFrame(
            {"context_id": [1], "initial_zone": [0]}
        ),
        logit_scale=0.5,
        update_plan_timings=True,
        use_shadow_prices=False,
        k=4,
    )
    ranked = (
        result.filter(pl.col("layer").is_in([0, 1]))
        .sort(["draw_id", "layer"])
        .group_by("draw_id", maintain_order=True)
        .agg(
            plan=pl.col("destination"),
            utility=pl.col("total_log_weight").first(),
        )
    )

    assert [tuple(plan) for plan in ranked["plan"]] == [
        (1, 2),
        (1, 1),
        (2, 2),
        (2, 1),
    ]
    assert ranked["utility"].to_list() == sorted(
        ranked["utility"].to_list(),
        reverse=True,
    )
    assert report["complete_plans"] == 4
    assert report["assignment_lattice"] == "4"

    best, best_report = sampler.search_ternary_top_k(
        steps=steps,
        initial_locations=pl.DataFrame(
            {"context_id": [1], "initial_zone": [0]}
        ),
        logit_scale=0.5,
        update_plan_timings=True,
        use_shadow_prices=False,
        k=1,
    )
    assert tuple(
        best.sort("layer").filter(pl.col("layer").is_in([0, 1]))[
            "destination"
        ]
    ) == (1, 2)
    assert best_report["incumbent_contexts"] == 1
    assert best_report["incumbent_children_considered"] > 0
    assert best_report["children_pruned_by_incumbent"] > 0


def test_heap_top_one_conditions_cross_home_anchor_exactly() -> None:
    zones = [0, 1, 2]
    od_costs = pl.DataFrame(
        {
            "origin": [origin for origin in zones for _ in zones],
            "destination": [
                destination for _ in zones for destination in zones
            ],
            "cost": [
                0.2 * abs(destination - origin)
                for origin in zones
                for destination in zones
            ],
            "time": [
                0.1 * abs(destination - origin)
                for origin in zones
                for destination in zones
            ],
        }
    )
    destination_inputs = pl.DataFrame(
        {
            "activity_id": [10, 10, 20, 20],
            "destination": [1, 2, 1, 2],
            "opportunity_capacity": [4.0, 1.0, 1.0, 3.0],
            "country_value_coefficient": [1.0] * 4,
            "saturation_utility": [1.0] * 4,
            "shadow_price": [0.0] * 4,
        }
    )
    steps = pl.DataFrame(
        {
            "context_id": [1] * 5,
            "layer": list(range(5)),
            "activity_id": [10, 0, 20, 10, 0],
            "anchor_id": [10, None, None, 10, None],
            "fixed_destination": [None, 0, None, None, 0],
            "departure_time": [8.0, 12.0, 13.0, 16.0, 18.0],
            "arrival_time": [8.5, 12.5, 13.5, 16.5, 18.5],
            "arrival_time_rigidity": [1.0, 1.0, 0.0, 1.0, 1.0],
            "next_departure_time": [12.0, 13.0, 16.0, 18.0, 19.0],
            "duration_per_person": [3.5, 0.5, 2.5, 1.5, 0.5],
            "value_of_time": [2.0, 0.0, 3.0, 2.0, 0.0],
            "mean_duration_per_person": [2.0, 1.0, 2.0, 2.0, 1.0],
            "min_activity_time": [0.5] * 5,
        }
    )
    sampler = DestinationSampler(
        od_costs=od_costs,
        destination_inputs=destination_inputs,
    )
    arguments = {
        "steps": steps,
        "initial_locations": pl.DataFrame(
            {"context_id": [1], "initial_zone": [0]}
        ),
        "logit_scale": 0.5,
        "update_plan_timings": True,
        "use_shadow_prices": False,
    }

    ranked, _ = sampler.search_ternary_top_k(**arguments, k=4)
    best, report = sampler.search_ternary_top_k(**arguments, k=1)

    assert best.sort("layer")["destination"].to_list() == (
        ranked.filter(pl.col("draw_id") == 1)
        .sort("layer")["destination"]
        .to_list()
    )
    assert best.sort("layer")["total_log_weight"][0] == pytest.approx(
        ranked.filter(pl.col("draw_id") == 1)
        .sort("layer")["total_log_weight"][0]
    )
    assert report["conditioned_anchor_contexts"] == 1
    assert (
        report["anchor_conditions_considered"]
        + report["anchor_conditions_pruned"]
        == 2
    )
    assert report["anchor_conditions_pruned"] > 0


def test_many_draws_follow_exact_backward_probabilities() -> None:
    steps, initial_locations, od_costs, destination_inputs = build_toy_inputs()
    result = sample_destination_sequences(
        steps=steps,
        initial_locations=initial_locations,
        od_costs=od_costs,
        destination_inputs=destination_inputs,
        logit_scale=1.0,
        update_plan_timings=True,
        use_shadow_prices=False,
        seed=7,
        n_draws=20_000,
    )

    first_steps = result.filter(result["layer"] == 0)
    observed_probability = (
        first_steps.filter(first_steps["destination"] == 1).height
        / first_steps.height
    )
    weights = toy_path_log_weights()
    maximum = max(weights.values())
    expected_probability = math.exp(weights[1] - maximum) / sum(
        math.exp(value - maximum) for value in weights.values()
    )

    assert abs(observed_probability - expected_probability) < 0.012


def test_reusable_sampler_matches_convenience_function() -> None:
    steps, initial_locations, od_costs, destination_inputs = build_toy_inputs()
    sampler = DestinationSampler(
        od_costs=od_costs,
        destination_inputs=destination_inputs,
    )
    reused = sampler.sample(
        steps=steps,
        initial_locations=initial_locations,
        logit_scale=1.0,
        update_plan_timings=True,
        use_shadow_prices=False,
        seed=42,
        n_draws=20,
    )
    reused_again = sampler.sample(
        steps=steps,
        initial_locations=initial_locations,
        logit_scale=1.0,
        update_plan_timings=True,
        use_shadow_prices=False,
        seed=42,
        n_draws=20,
    )
    direct = sample_destination_sequences(
        steps=steps,
        initial_locations=initial_locations,
        od_costs=od_costs,
        destination_inputs=destination_inputs,
        logit_scale=1.0,
        update_plan_timings=True,
        use_shadow_prices=False,
        seed=42,
        n_draws=20,
    )

    assert reused.equals(direct)
    assert reused_again.equals(direct)


def test_shadow_price_and_reference_duration_utility() -> None:
    steps, initial_locations, od_costs, destination_inputs = build_toy_inputs()
    destination_inputs = destination_inputs.with_columns(
        country_value_coefficient=destination_inputs[
            "destination"
        ].replace_strict(
            {1: 1.1, 2: 0.8},
            return_dtype=pl.Float64,
        ),
        shadow_price=destination_inputs["destination"].replace_strict(
            {1: 0.2, 2: -0.3},
            return_dtype=pl.Float64,
        ),
    )
    result = sample_destination_sequences(
        steps=steps,
        initial_locations=initial_locations,
        od_costs=od_costs,
        destination_inputs=destination_inputs,
        logit_scale=0.7,
        update_plan_timings=False,
        use_shadow_prices=True,
        seed=11,
        n_draws=100,
    )

    first_steps = result.filter(result["layer"] == 0)
    for row in first_steps.iter_rows(named=True):
        destination = row["destination"]
        if destination == 1:
            capacity, country_coefficient, shadow_price, cost = 2.0, 1.1, 0.2, 1.0
        else:
            capacity, country_coefficient, shadow_price, cost = 1.0, 0.8, -0.3, 1.4
        activity_coefficient = country_coefficient * 1.0 + shadow_price
        activity_utility = activity_coefficient * 8.0 * math.log(8.0 / 4.0)
        expected = math.log(capacity) + 0.7 * (activity_utility - cost)
        assert math.isclose(
            row["local_log_weight"],
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        )


def test_repeated_anchor_visits_share_one_destination() -> None:
    zones = [0, 1, 2]
    od_costs = pl.DataFrame(
        {
            "origin": [origin for origin in zones for _ in zones],
            "destination": [destination for _ in zones for destination in zones],
            "cost": [
                0.0 if origin == destination else 1.0
                for origin in zones
                for destination in zones
            ],
            "time": [
                0.0 if origin == destination else 0.25
                for origin in zones
                for destination in zones
            ],
        }
    )
    destination_inputs = pl.DataFrame(
        {
            "activity_id": [10, 10],
            "destination": [1, 2],
            "opportunity_capacity": [5.0, 1.0],
            "country_value_coefficient": [1.0, 1.0],
            "saturation_utility": [1.0, 1.0],
            "shadow_price": [0.0, 0.0],
        }
    )
    steps = pl.DataFrame(
        {
            "context_id": [1, 1, 1, 1],
            "layer": [0, 1, 2, 3],
            "activity_id": [10, 0, 10, 0],
            "anchor_id": [10, None, 10, None],
            "fixed_destination": [None, 0, None, 0],
            "departure_time": [8.0, 12.0, 13.0, 17.0],
            "next_departure_time": [12.0, 13.0, 17.0, 18.0],
            "duration_per_person": [3.5, 0.75, 3.5, 0.75],
            "value_of_time": [1.0, 0.0, 1.0, 0.0],
            "mean_duration_per_person": [3.0, 1.0, 3.0, 1.0],
            "min_activity_time": [1.0, 1.0, 1.0, 1.0],
        }
    )
    initial_locations = pl.DataFrame(
        {"context_id": [1], "initial_zone": [0]}
    )

    result = sample_destination_sequences(
        steps=steps,
        initial_locations=initial_locations,
        od_costs=od_costs,
        destination_inputs=destination_inputs,
        logit_scale=1.0,
        update_plan_timings=True,
        use_shadow_prices=False,
        seed=19,
        n_draws=1_000,
    )

    for draw in rows_by_draw(result):
        assert draw["destination"][0] == draw["destination"][2]
        # Capacity is an attribute of the workplace choice, not of each visit.
        first_capacity_term = (
            math.log(5.0) if draw["destination"][0] == 1 else 0.0
        )
        assert math.isclose(
            draw["local_log_weight"][0]
            - draw["local_log_weight"][2],
            first_capacity_term,
            rel_tol=0.0,
            abs_tol=1e-12,
        )


def test_two_anchor_types_each_keep_their_own_destination() -> None:
    zones = [0, 1, 2]
    od_costs = pl.DataFrame(
        {
            "origin": [origin for origin in zones for _ in zones],
            "destination": [destination for _ in zones for destination in zones],
            "cost": [0.0] * 9,
            "time": [0.0] * 9,
        }
    )
    destination_inputs = pl.DataFrame(
        {
            "activity_id": [10, 10, 20, 20],
            "destination": [1, 2, 1, 2],
            "opportunity_capacity": [1.0, 1.0, 1.0, 1.0],
            "country_value_coefficient": [1.0] * 4,
            "saturation_utility": [1.0] * 4,
            "shadow_price": [0.0] * 4,
        }
    )
    steps = pl.DataFrame(
        {
            "context_id": [1] * 5,
            "layer": list(range(5)),
            "activity_id": [10, 20, 10, 20, 0],
            "anchor_id": [10, 20, 10, 20, None],
            "fixed_destination": [None, None, None, None, 0],
            "departure_time": [8.0, 10.0, 12.0, 14.0, 16.0],
            "next_departure_time": [10.0, 12.0, 14.0, 16.0, 17.0],
            "duration_per_person": [2.0] * 5,
            "value_of_time": [0.0] * 5,
            "mean_duration_per_person": [1.0] * 5,
            "min_activity_time": [1.0] * 5,
        }
    )
    result = sample_destination_sequences(
        steps=steps,
        initial_locations=pl.DataFrame(
            {"context_id": [1], "initial_zone": [0]}
        ),
        od_costs=od_costs,
        destination_inputs=destination_inputs,
        logit_scale=1.0,
        update_plan_timings=True,
        use_shadow_prices=False,
        seed=29,
        n_draws=100,
    )

    for draw in rows_by_draw(result):
        assert draw["destination"][0] == draw["destination"][2]
        assert draw["destination"][1] == draw["destination"][3]


def test_complete_chain_samples_flexible_visit_between_repeated_anchors() -> None:
    zones = [0, 1, 2, 3]
    od_costs = pl.DataFrame(
        {
            "origin": [origin for origin in zones for _ in zones],
            "destination": [
                destination for _ in zones for destination in zones
            ],
            "cost": [
                0.0 if origin == destination else 1.0
                for origin in zones
                for destination in zones
            ],
            "time": [0.0] * 16,
        }
    )
    destination_inputs = pl.DataFrame(
        {
            "activity_id": [10, 10, 30, 30],
            "destination": [1, 2, 2, 3],
            "opportunity_capacity": [3.0, 1.0, 1.0, 2.0],
            "country_value_coefficient": [1.0] * 4,
            "saturation_utility": [1.0] * 4,
            "shadow_price": [0.0] * 4,
        }
    )
    steps = pl.DataFrame(
        {
            "context_id": [1] * 4,
            "layer": [0, 1, 2, 3],
            "activity_id": [10, 30, 10, 0],
            "anchor_id": [10, None, 10, None],
            "fixed_destination": [None, None, None, 0],
            "departure_time": [8.0, 12.0, 14.0, 17.0],
            "next_departure_time": [12.0, 14.0, 17.0, 18.0],
            "duration_per_person": [4.0, 2.0, 3.0, 1.0],
            "value_of_time": [0.0] * 4,
            "mean_duration_per_person": [1.0] * 4,
            "min_activity_time": [1.0] * 4,
        }
    )

    result = sample_destination_sequences(
        steps=steps,
        initial_locations=pl.DataFrame(
            {"context_id": [1], "initial_zone": [0]}
        ),
        od_costs=od_costs,
        destination_inputs=destination_inputs,
        logit_scale=1.0,
        update_plan_timings=True,
        use_shadow_prices=False,
        seed=41,
        n_draws=500,
    )

    assert set(result.filter(pl.col("layer") == 1)["destination"]) == {2, 3}
    for draw in rows_by_draw(result):
        assert draw["destination"][0] == draw["destination"][2]
        assert draw["destination"][1] in (2, 3)
        assert draw["destination"][3] == 0
