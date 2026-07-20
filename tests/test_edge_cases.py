from __future__ import annotations

import polars as pl
import pytest

from mobility_destination_sequence_sampler._core import (
    ExperimentalDestinationSampler as DestinationSampler,
    sample_destination_sequences,
)

from conftest import build_toy_inputs


def test_rejects_non_consecutive_layers() -> None:
    steps, initial_locations, od_costs, destination_inputs = build_toy_inputs()
    steps = steps.with_columns(
        pl.when(pl.col("layer") == 1)
        .then(2)
        .otherwise(pl.col("layer"))
        .alias("layer")
    )

    with pytest.raises(ValueError, match="layers must be consecutive"):
        sample_destination_sequences(
            steps=steps,
            initial_locations=initial_locations,
            od_costs=od_costs,
            destination_inputs=destination_inputs,
            logit_scale=1.0,
            update_plan_timings=True,
            use_shadow_prices=False,
            seed=1,
        )


def test_reports_context_without_feasible_sequence() -> None:
    steps, initial_locations, od_costs, destination_inputs = build_toy_inputs()
    destination_inputs = destination_inputs.with_columns(
        opportunity_capacity=pl.lit(0.0)
    )

    with pytest.raises(ValueError, match="no feasible destination sequence"):
        sample_destination_sequences(
            steps=steps,
            initial_locations=initial_locations,
            od_costs=od_costs,
            destination_inputs=destination_inputs,
            logit_scale=1.0,
            update_plan_timings=True,
            use_shadow_prices=False,
            seed=1,
        )


def test_can_skip_context_without_feasible_sequence() -> None:
    steps, initial_locations, od_costs, destination_inputs = build_toy_inputs()
    destination_inputs = destination_inputs.with_columns(
        opportunity_capacity=pl.lit(0.0)
    )

    result = sample_destination_sequences(
        steps=steps,
        initial_locations=initial_locations,
        od_costs=od_costs,
        destination_inputs=destination_inputs,
        logit_scale=1.0,
        update_plan_timings=True,
        use_shadow_prices=False,
        seed=1,
        skip_infeasible=True,
    )

    assert result.is_empty()


def test_reference_can_skip_context_without_feasible_sequence() -> None:
    steps, initial_locations, od_costs, destination_inputs = build_toy_inputs()
    steps = steps.with_columns(
        arrival_time=pl.col("departure_time") + 0.5,
        arrival_time_rigidity=pl.lit(0.5),
        departure_time_rigidity=pl.lit(0.5),
    )
    destination_inputs = destination_inputs.with_columns(
        opportunity_capacity=pl.lit(0.0)
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
        skip_infeasible=True,
    )

    assert result.is_empty()


def test_sparse_od_graph_uses_the_same_sampling_contract() -> None:
    steps, initial_locations, _, destination_inputs = build_toy_inputs()
    od_costs = pl.DataFrame(
        {
            "origin": [0, 1] + list(range(2, 10)),
            "destination": [1, 0] + list(range(2, 10)),
            "cost": [1.0, 1.0] + [0.0] * 8,
            "time": [0.25, 0.25] + [0.0] * 8,
        }
    )
    destination_inputs = destination_inputs.filter(
        pl.col("destination") == 1
    )

    result = sample_destination_sequences(
        steps=steps,
        initial_locations=initial_locations,
        od_costs=od_costs,
        destination_inputs=destination_inputs,
        logit_scale=1.0,
        update_plan_timings=True,
        use_shadow_prices=False,
        seed=1,
    )

    assert result["destination"].to_list() == [1, 0]


def test_reports_complete_chain_with_a_cyclic_factor_graph() -> None:
    zones = [0, 1, 2]
    od_costs = pl.DataFrame(
        {
            "origin": [origin for origin in zones for _ in zones],
            "destination": [
                destination for _ in zones for destination in zones
            ],
            "cost": [0.0] * 9,
            "time": [0.0] * 9,
        }
    )
    destination_inputs = pl.DataFrame(
        {
            "activity_id": [10, 10, 20, 20, 30, 30],
            "destination": [1, 2, 1, 2, 1, 2],
            "opportunity_capacity": [1.0] * 6,
            "country_value_coefficient": [1.0] * 6,
            "saturation_utility": [1.0] * 6,
            "shadow_price": [0.0] * 6,
        }
    )
    steps = pl.DataFrame(
        {
            "context_id": [1] * 4,
            "layer": [0, 1, 2, 3],
            "activity_id": [10, 20, 30, 10],
            "anchor_id": [10, None, None, 10],
            "fixed_destination": [None, None, None, None],
            "departure_time": [8.0, 10.0, 12.0, 14.0],
            "next_departure_time": [10.0, 12.0, 14.0, 16.0],
            "duration_per_person": [2.0] * 4,
            "value_of_time": [0.0] * 4,
            "mean_duration_per_person": [1.0] * 4,
            "min_activity_time": [1.0] * 4,
        }
    )
    initial_locations = pl.DataFrame(
        {"context_id": [1], "initial_zone": [0]}
    )

    with pytest.raises(ValueError, match="cyclic destination factor graph"):
        sample_destination_sequences(
            steps=steps,
            initial_locations=initial_locations,
            od_costs=od_costs,
            destination_inputs=destination_inputs,
            logit_scale=1.0,
            update_plan_timings=True,
            use_shadow_prices=False,
            seed=1,
        )
