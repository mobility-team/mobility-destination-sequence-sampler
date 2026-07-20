from __future__ import annotations

import polars as pl
import pytest

from mobility_destination_sequence_sampler._core import (
    ExperimentalDestinationSampler as DestinationSampler,
)

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


def test_particle_sampler_is_reproducible_and_scores_complete_plans() -> None:
    steps, initial_locations, od_costs, destination_inputs = reference_steps()
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
        "seed": 7,
        "n_particles": 16,
        "candidate_count": 2,
        "n_draws": 2,
    }

    result, report = sampler.sample_particles(**arguments)
    repeated, repeated_report = sampler.sample_particles(**arguments)

    assert result.equals(repeated)
    assert report == repeated_report
    assert result.group_by("draw_id").len()["len"].to_list() == [2, 2]
    assert set(result.filter(pl.col("layer") == 1)["destination"]) == {0}
    assert report["candidate_evaluations"] <= 16 * 2 * 8
    assert report["completed_particles"] == 16
    assert report["mean_effective_sample_size"] > 0.0
    assert result["proposal_log_probability"].is_finite().all()
    assert result["importance_log_weight"].is_finite().all()


def test_particle_sampler_requires_reference_timing_columns() -> None:
    steps, initial_locations, od_costs, destination_inputs = build_toy_inputs()
    sampler = DestinationSampler(
        od_costs=od_costs,
        destination_inputs=destination_inputs,
    )

    with pytest.raises(Exception, match="arrival_time"):
        sampler.sample_particles(
            steps=steps,
            initial_locations=initial_locations,
            logit_scale=1.0,
            update_plan_timings=True,
            use_shadow_prices=False,
            seed=7,
        )


def test_particle_sampler_pins_home_and_repeated_anchor() -> None:
    zones = [0, 1, 2]
    od_costs = pl.DataFrame(
        {
            "origin": [origin for origin in zones for _ in zones],
            "destination": [destination for _ in zones for destination in zones],
            "cost": [0.0 if origin == destination else 0.2 for origin in zones for destination in zones],
            "time": [0.0 if origin == destination else 0.1 for origin in zones for destination in zones],
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
            "context_id": [1] * 4,
            "layer": list(range(4)),
            "activity_id": [10, 0, 10, 0],
            "anchor_id": [99, None, 99, None],
            "fixed_destination": [None, 0, None, 0],
            "departure_time": [8.0, 12.0, 13.0, 17.0],
            "arrival_time": [8.5, 12.5, 13.5, 17.5],
            "arrival_time_rigidity": [0.5] * 4,
            "departure_time_rigidity": [0.5] * 4,
            "next_departure_time": [12.0, 13.0, 17.0, 18.0],
            "duration_per_person": [3.0, 0.5, 3.0, 0.5],
            "value_of_time": [1.0, 0.0, 1.0, 0.0],
            "mean_duration_per_person": [2.0, 1.0, 2.0, 1.0],
            "min_activity_time": [1.0] * 4,
        }
    )
    sampler = DestinationSampler(
        od_costs=od_costs,
        destination_inputs=destination_inputs,
    )
    result, _ = sampler.sample_particles(
        steps=steps,
        initial_locations=pl.DataFrame({"context_id": [1], "initial_zone": [0]}),
        logit_scale=1.0,
        update_plan_timings=True,
        use_shadow_prices=False,
        seed=11,
        n_particles=24,
        candidate_count=2,
        n_draws=4,
    )

    for draw_id in result["draw_id"].unique():
        destinations = result.filter(pl.col("draw_id") == draw_id).sort("layer")["destination"]
        assert destinations[0] == destinations[2]
        assert destinations[1] == destinations[3] == 0


def test_particle_sampler_eliminates_invalid_duration_without_repair() -> None:
    steps, initial_locations, od_costs, destination_inputs = reference_steps()
    steps = steps.with_columns(
        pl.when(pl.col("layer") == 0)
        .then(8.0)
        .otherwise(pl.col("next_departure_time"))
        .alias("next_departure_time")
    )
    sampler = DestinationSampler(
        od_costs=od_costs,
        destination_inputs=destination_inputs,
    )

    result, report = sampler.sample_particles(
        steps=steps,
        initial_locations=initial_locations,
        logit_scale=1.0,
        update_plan_timings=True,
        use_shadow_prices=False,
        seed=7,
        skip_infeasible=True,
    )

    assert result.is_empty()
    assert report["infeasible_contexts"] == 1


def test_particle_sampler_keeps_flexible_terminal_home() -> None:
    steps, initial_locations, od_costs, destination_inputs = reference_steps()
    steps = steps.with_columns(
        pl.when(pl.col("layer") == 1)
        .then(0.0)
        .otherwise(pl.col("next_departure_time"))
        .alias("next_departure_time")
    )
    sampler = DestinationSampler(
        od_costs=od_costs,
        destination_inputs=destination_inputs,
    )

    result, report = sampler.sample_particles(
        steps=steps,
        initial_locations=initial_locations,
        logit_scale=1.0,
        update_plan_timings=True,
        use_shadow_prices=False,
        seed=7,
        n_particles=8,
        candidate_count=2,
        n_draws=2,
    )

    assert result.height == 4
    assert report["completed_particles"] == 8


def test_particle_sampler_rejects_transition_that_invalidates_previous_activity() -> None:
    zones = [0, 1]
    od_costs = pl.DataFrame(
        {
            "origin": [origin for origin in zones for _ in zones],
            "destination": [destination for _ in zones for destination in zones],
            "cost": [0.0] * 4,
            "time": [0.0, 1.0, 0.0, 0.0],
        }
    )
    destination_inputs = pl.DataFrame(
        {
            "activity_id": [10, 20],
            "destination": [1, 1],
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
            "departure_time": [8.0, 8.5, 10.0],
            "arrival_time": [8.5, 9.0, 10.5],
            "arrival_time_rigidity": [0.5, 0.5, 0.0],
            "departure_time_rigidity": [0.5, 0.5, 0.5],
            "next_departure_time": [12.0, 10.0, 11.0],
            "duration_per_person": [2.0, 1.0, 0.0],
            "value_of_time": [1.0, 1.0, 0.0],
            "mean_duration_per_person": [1.0, 1.0, 1.0],
            "min_activity_time": [0.5, 0.5, 0.5],
        }
    )
    sampler = DestinationSampler(
        od_costs=od_costs,
        destination_inputs=destination_inputs,
    )

    result, report = sampler.sample_particles(
        steps=steps,
        initial_locations=pl.DataFrame({"context_id": [1], "initial_zone": [0]}),
        logit_scale=1.0,
        update_plan_timings=True,
        use_shadow_prices=False,
        seed=3,
        n_particles=8,
        candidate_count=2,
        n_draws=1,
        skip_infeasible=True,
    )

    assert result.is_empty()
    context_report = report["context_reports"][0]
    assert context_report["failure_reason"] == "no_locally_feasible_candidate"
    assert context_report["first_failure_layer"] == 1


def test_particle_sampler_retry_recovers_alternate_previous_destination() -> None:
    zones = [0, 1, 2]
    od_costs = pl.DataFrame(
        {
            "origin": [origin for origin in zones for _ in zones],
            "destination": [destination for _ in zones for destination in zones],
            "cost": [0.0] * 9,
            # Destination 1 makes the next departure leave no time at the
            # first activity; destination 2 remains feasible.
            "time": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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
            "context_id": [1, 1, 1],
            "layer": [0, 1, 2],
            "activity_id": [10, 20, 0],
            "anchor_id": [None, None, None],
            "fixed_destination": [None, 2, 0],
            "departure_time": [8.0, 8.5, 10.0],
            "arrival_time": [8.5, 9.0, 10.5],
            "arrival_time_rigidity": [0.5, 0.5, 0.0],
            "departure_time_rigidity": [0.5, 0.5, 0.5],
            "next_departure_time": [12.0, 10.0, 11.0],
            "duration_per_person": [2.0, 1.0, 0.0],
            "value_of_time": [1.0, 1.0, 0.0],
            "mean_duration_per_person": [1.0, 1.0, 1.0],
            "min_activity_time": [0.5, 0.5, 0.5],
        }
    )
    sampler = DestinationSampler(
        od_costs=od_costs,
        destination_inputs=destination_inputs,
    )
    arguments = {
        "steps": steps,
        "initial_locations": pl.DataFrame({"context_id": [1], "initial_zone": [0]}),
        "logit_scale": 1.0,
        "update_plan_timings": True,
        "use_shadow_prices": False,
        "seed": 0,
        "n_particles": 1,
        "candidate_count": 2,
        "n_draws": 1,
        "skip_infeasible": True,
    }

    without_retry, no_retry_report = sampler.sample_particles(
        **arguments, max_retries=0
    )
    recovered, retry_report = sampler.sample_particles(
        **arguments, max_retries=2
    )

    assert without_retry.is_empty()
    assert no_retry_report["retry_attempts"] == 0
    assert recovered.height == 3
    assert retry_report["retry_attempts"] == 1
    assert retry_report["recovered_contexts"] == 1
    assert retry_report["context_reports"][0]["recovered_by_retry"] is True

