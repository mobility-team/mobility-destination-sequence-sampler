from __future__ import annotations

import argparse
import random
import threading
import time

import polars as pl
import psutil

from mobility_destination_sequence_sampler._core import (
    ExperimentalDestinationSampler as DestinationSampler,
)


def build_od_costs(
    *,
    n_zones: int,
    out_degree: int,
    seed: int,
) -> pl.DataFrame:
    rng = random.Random(seed)
    rows: dict[str, list[int | float]] = {
        "origin": [],
        "destination": [],
        "cost": [],
        "time": [],
    }
    for origin in range(n_zones):
        destinations = rng.sample(range(n_zones), min(out_degree, n_zones))
        if origin not in destinations:
            destinations[-1] = origin
        for destination in sorted(set(destinations)):
            ring_distance = min(
                (destination - origin) % n_zones,
                (origin - destination) % n_zones,
            )
            rows["origin"].append(origin)
            rows["destination"].append(destination)
            rows["cost"].append(0.25 + ring_distance / 35.0)
            rows["time"].append(0.05 + ring_distance / 80.0)
    return pl.DataFrame(rows)


def build_destination_inputs(
    *,
    n_zones: int,
    n_activities: int,
) -> pl.DataFrame:
    rows: dict[str, list[int | float]] = {
        "activity_id": [],
        "destination": [],
        "opportunity_capacity": [],
        "country_value_coefficient": [],
        "saturation_utility": [],
        "shadow_price": [],
    }
    for activity_id in range(n_activities):
        for destination in range(n_zones):
            rows["activity_id"].append(activity_id)
            rows["destination"].append(destination)
            rows["opportunity_capacity"].append(
                1.0 + ((destination * 17 + activity_id * 13) % 100)
            )
            rows["country_value_coefficient"].append(
                0.9 + 0.05 * (destination % 3)
            )
            rows["saturation_utility"].append(
                0.7 + 0.3 * ((destination * 7) % 11) / 10.0
            )
            rows["shadow_price"].append(
                -0.2 + 0.4 * ((destination * 5) % 13) / 12.0
            )
    return pl.DataFrame(rows)


def build_contexts(
    *,
    n_contexts: int,
    n_layers: int,
    n_zones: int,
    n_activities: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    rows: dict[str, list[int | float | None]] = {
        "context_id": [],
        "layer": [],
        "activity_id": [],
        "anchor_id": [],
        "fixed_destination": [],
        "departure_time": [],
        "next_departure_time": [],
        "duration_per_person": [],
        "value_of_time": [],
        "mean_duration_per_person": [],
        "min_activity_time": [],
    }
    initial_context_id = []
    initial_zone = []
    for context_id in range(n_contexts):
        home = context_id % n_zones
        initial_context_id.append(context_id)
        initial_zone.append(home)
        for layer in range(n_layers):
            is_terminal = layer == n_layers - 1
            rows["context_id"].append(context_id)
            rows["layer"].append(layer)
            rows["activity_id"].append(layer % n_activities)
            rows["anchor_id"].append(None)
            rows["fixed_destination"].append(home if is_terminal else None)
            rows["departure_time"].append(6.0 + layer * 2.5)
            rows["next_departure_time"].append(9.0 + layer * 2.5)
            rows["duration_per_person"].append(2.5)
            rows["value_of_time"].append(1.5 + 0.1 * (context_id % 5))
            rows["mean_duration_per_person"].append(2.0)
            rows["min_activity_time"].append(0.5)
    return (
        pl.DataFrame(rows),
        pl.DataFrame(
            {"context_id": initial_context_id, "initial_zone": initial_zone}
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark exact backward-forward destination sampling."
    )
    parser.add_argument("--n-zones", type=int, default=256)
    parser.add_argument("--out-degree", type=int, default=96)
    parser.add_argument("--n-contexts", type=int, default=200)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-activities", type=int, default=4)
    parser.add_argument("--n-draws", type=int, default=3)
    parser.add_argument("--n-threads", type=int, default=None)
    parser.add_argument("--track-memory", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    od_costs = build_od_costs(
        n_zones=args.n_zones,
        out_degree=args.out_degree,
        seed=args.seed,
    )
    destination_inputs = build_destination_inputs(
        n_zones=args.n_zones,
        n_activities=args.n_activities,
    )
    steps, initial_locations = build_contexts(
        n_contexts=args.n_contexts,
        n_layers=args.n_layers,
        n_zones=args.n_zones,
        n_activities=args.n_activities,
    )

    build_started = time.perf_counter()
    sampler = DestinationSampler(
        od_costs=od_costs,
        destination_inputs=destination_inputs,
    )
    build_elapsed = time.perf_counter() - build_started

    peak_rss = [psutil.Process().memory_info().rss]
    stop_memory_poll = threading.Event()

    def poll_memory() -> None:
        process = psutil.Process()
        while not stop_memory_poll.wait(0.01):
            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)

    memory_thread = None
    if args.track_memory:
        memory_thread = threading.Thread(target=poll_memory, daemon=True)
        memory_thread.start()

    sample_started = time.perf_counter()
    result = sampler.sample(
        steps=steps,
        initial_locations=initial_locations,
        logit_scale=1.0,
        update_plan_timings=True,
        use_shadow_prices=False,
        seed=args.seed,
        n_draws=args.n_draws,
        n_threads=args.n_threads,
    )
    sample_elapsed = time.perf_counter() - sample_started
    if memory_thread is not None:
        stop_memory_poll.set()
        memory_thread.join()
        peak_rss[0] = max(peak_rss[0], psutil.Process().memory_info().rss)
    edge_evaluations = (
        args.n_contexts
        * max(args.n_layers - 1, 0)
        * od_costs.height
    )

    print("Synthetic destination sampler benchmark")
    print(f"zones: {args.n_zones:,}")
    print(f"OD edges: {od_costs.height:,}")
    print(f"contexts: {args.n_contexts:,}")
    print(f"layers per context: {args.n_layers}")
    print(f"draws per context: {args.n_draws}")
    print(f"sampled output rows: {result.height:,}")
    print(f"index build seconds: {build_elapsed:.3f}")
    print(f"sample seconds: {sample_elapsed:.3f}")
    print(f"total seconds: {build_elapsed + sample_elapsed:.3f}")
    if args.track_memory:
        print(f"peak process memory: {peak_rss[0] / 1024**2:,.1f} MiB")
    print(f"estimated backward edge evaluations: {edge_evaluations:,}")
    print(
        "estimated backward edge evaluations per second: "
        f"{edge_evaluations / sample_elapsed:,.0f}"
    )


if __name__ == "__main__":
    main()
