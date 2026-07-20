from __future__ import annotations

import argparse
import time

import polars as pl

from mobility_destination_sequence_sampler._core import (
    ExperimentalDestinationSampler as DestinationSampler,
)


def build_case(
    *,
    n_zones: int,
    n_variable_layers: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    zones = range(n_zones)
    od_costs = pl.DataFrame(
        {
            "origin": [origin for origin in zones for _ in zones],
            "destination": [
                destination for _ in zones for destination in zones
            ],
            "cost": [
                0.25
                * min(
                    abs(destination - origin),
                    n_zones - abs(destination - origin),
                )
                for origin in zones
                for destination in zones
            ],
            "time": [
                0.12
                * min(
                    abs(destination - origin),
                    n_zones - abs(destination - origin),
                )
                for origin in zones
                for destination in zones
            ],
        }
    )
    activities = range(1, n_variable_layers + 1)
    destination_inputs = pl.DataFrame(
        {
            "activity_id": [
                activity for activity in activities for _ in zones
            ],
            "destination": [
                destination
                for _ in activities
                for destination in zones
            ],
            "opportunity_capacity": [
                1.0 + ((activity * 13 + destination * 17) % 40)
                for activity in activities
                for destination in zones
            ],
            "country_value_coefficient": [
                1.0 for _ in activities for _ in zones
            ],
            "saturation_utility": [
                1.0 for _ in activities for _ in zones
            ],
            "shadow_price": [0.0 for _ in activities for _ in zones],
        }
    )

    rows = []
    for layer in range(n_variable_layers + 1):
        is_home = layer == n_variable_layers
        departure_time = 7.0 + 3.0 * layer
        rows.append(
            (
                1,
                layer,
                0 if is_home else layer + 1,
                None,
                0 if is_home else None,
                departure_time,
                departure_time + 0.5,
                departure_time + 2.5,
                2.0,
                2.0,
                2.0,
                0.5,
                0.7,
            )
        )
    steps = pl.DataFrame(
        rows,
        schema=[
            "context_id",
            "layer",
            "activity_id",
            "anchor_id",
            "fixed_destination",
            "departure_time",
            "arrival_time",
            "next_departure_time",
            "duration_per_person",
            "value_of_time",
            "mean_duration_per_person",
            "min_activity_time",
            "arrival_time_rigidity",
        ],
        orient="row",
    )
    initial_locations = pl.DataFrame(
        {"context_id": [1], "initial_zone": [0]}
    )
    return od_costs, destination_inputs, steps, initial_locations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark exact best-first complete-plan search."
    )
    parser.add_argument("--n-zones", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument("--n-variable-layers", type=int, default=4)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--max-states", type=int, default=10_000_000)
    args = parser.parse_args()

    print("Exact heap destination search benchmark")
    for n_zones in args.n_zones:
        od_costs, destination_inputs, steps, initial_locations = build_case(
            n_zones=n_zones,
            n_variable_layers=args.n_variable_layers,
        )
        started = time.perf_counter()
        sampler = DestinationSampler(
            od_costs=od_costs,
            destination_inputs=destination_inputs,
        )
        build_seconds = time.perf_counter() - started

        started = time.perf_counter()
        result, report = sampler.search_ternary_top_k(
            steps=steps,
            initial_locations=initial_locations,
            logit_scale=1.0,
            update_plan_timings=True,
            use_shadow_prices=False,
            k=args.k,
            max_states=args.max_states,
        )
        search_seconds = time.perf_counter() - started
        lattice = n_zones**args.n_variable_layers
        print(
            f"zones={n_zones:>4} lattice={lattice:>12,} "
            f"build={build_seconds:>8.4f}s search={search_seconds:>8.4f}s "
            f"popped={report['states_popped']:>10,} "
            f"children={report['children_considered']:>12,} "
            f"heap={report['maximum_heap_size']:>10,} "
            f"rows={result.height}"
        )


if __name__ == "__main__":
    main()
