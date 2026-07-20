from __future__ import annotations

import math
from collections.abc import Iterator

import polars as pl


def build_toy_inputs() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    # One flexible activity followed by a fixed return home. Destination 2 has
    # a worse first leg but a better return, so the backward value matters.
    od_costs = pl.DataFrame(
        {
            "origin": [0, 0, 1, 1, 2, 2],
            "destination": [1, 2, 0, 2, 0, 1],
            "cost": [1.0, 1.4, 1.0, 0.4, 0.3, 0.4],
            "time": [0.25, 0.45, 0.25, 0.10, 0.10, 0.10],
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
            "context_id": [100, 100],
            "layer": [0, 1],
            "activity_id": [10, 0],
            "anchor_id": [None, None],
            "fixed_destination": [None, 0],
            "departure_time": [8.0, 17.0],
            "next_departure_time": [17.0, 18.0],
            "duration_per_person": [8.0, 1.0],
            "value_of_time": [1.0, 0.0],
            "mean_duration_per_person": [8.0, 1.0],
            "min_activity_time": [4.0, 1.0],
        }
    )
    initial_locations = pl.DataFrame(
        {"context_id": [100], "initial_zone": [0]}
    )
    return steps, initial_locations, od_costs, destination_inputs


def toy_path_log_weights() -> dict[int, float]:
    # This mirrors the model formula without depending on the Rust result.
    values = {}
    for destination, capacity, outbound_cost, outbound_time, return_cost in [
        (1, 2.0, 1.0, 0.25, 1.0),
        (2, 1.0, 1.4, 0.45, 0.3),
    ]:
        duration = 17.0 - 8.0 - outbound_time
        activity_utility = 8.0 * max(math.log(duration / 4.0), 0.0)
        outbound = math.log(capacity) + activity_utility - outbound_cost
        returned_home = -return_cost
        values[destination] = outbound + returned_home
    return values


def rows_by_draw(frame: pl.DataFrame) -> Iterator[pl.DataFrame]:
    for draw_id in frame["draw_id"].unique().sort():
        yield frame.filter(pl.col("draw_id") == draw_id).sort("layer")
