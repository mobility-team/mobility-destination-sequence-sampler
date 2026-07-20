from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl

from mobility_destination_sequence_sampler._core import (
    ExperimentalDestinationSampler as DestinationSampler,
)
from experiments.benchmarks.perf_grand_geneve_cache import (
    DEFAULT_GROUP_DAY_TRIPS_FOLDER,
    LOGIT_SCALE,
    prepare_complete_contexts,
    prepare_destination_inputs,
    prepare_od_costs,
    resolve_snapshot_files,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark exact heap search on cached Grand Genève inputs."
    )
    parser.add_argument(
        "--group-day-trips-folder",
        type=Path,
        default=DEFAULT_GROUP_DAY_TRIPS_FOLDER,
    )
    parser.add_argument("--n-contexts", type=int, default=10)
    parser.add_argument("--context-id", type=int, default=None)
    parser.add_argument("--exclude-context-ids", nargs="*", type=int, default=[])
    parser.add_argument("--minimum-variable-layers", type=int, default=2)
    parser.add_argument("--maximum-variable-layers", type=int, default=3)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--max-states", type=int, default=2_000_000)
    parser.add_argument("--n-threads", type=int, default=None)
    parser.add_argument("--per-context", action="store_true")
    parser.add_argument("--quiet-contexts", action="store_true")
    parser.add_argument("--seed", type=int, default=37)
    args = parser.parse_args()

    files = resolve_snapshot_files(args.group_day_trips_folder)
    started = time.perf_counter()
    od_costs = prepare_od_costs(
        files["transport_costs"],
        files["demand_groups"],
    )
    destination_inputs = prepare_destination_inputs(
        files["destination_saturation"],
        files["demand_groups"],
    )
    steps, initial_locations, _ = prepare_complete_contexts(
        activity_sequences_path=files["activity_sequences"],
        survey_plan_steps_path=files["survey_plan_steps"],
        demand_groups_path=files["demand_groups"],
        activity_dur_path=files["activity_dur"],
    )
    preparation_seconds = time.perf_counter() - started

    if args.context_id is None:
        selected_ids = (
            steps.group_by("context_id")
            .agg(
                variable_layers=pl.col("fixed_destination").is_null().sum(),
                layers=pl.len(),
            )
            .filter(
                pl.col("variable_layers").is_between(
                    args.minimum_variable_layers,
                    args.maximum_variable_layers,
                )
            )
            .with_columns(order=pl.col("context_id").hash(seed=args.seed))
            .sort("order")
            .head(args.n_contexts)
            .select("context_id")
        )
    else:
        selected_ids = pl.DataFrame(
            {"context_id": [args.context_id]},
            schema={"context_id": pl.UInt64},
        )
    steps = steps.join(selected_ids, on="context_id", how="semi")
    initial_locations = initial_locations.join(
        selected_ids,
        on="context_id",
        how="semi",
    )
    if args.exclude_context_ids:
        steps = steps.filter(
            ~pl.col("context_id").is_in(args.exclude_context_ids)
        )
        initial_locations = initial_locations.filter(
            ~pl.col("context_id").is_in(args.exclude_context_ids)
        )

    started = time.perf_counter()
    sampler = DestinationSampler(
        od_costs=od_costs,
        destination_inputs=destination_inputs,
    )
    build_seconds = time.perf_counter() - started
    search_arguments = {
        "logit_scale": LOGIT_SCALE,
        "update_plan_timings": True,
        "use_shadow_prices": True,
        "k": args.k,
        "max_states": args.max_states,
        "n_threads": args.n_threads,
        "skip_infeasible": True,
    }
    started = time.perf_counter()
    if args.per_context:
        results = []
        reports = []
        failed_contexts = []
        context_timings = []
        for context_id in initial_locations["context_id"]:
            context_started = time.perf_counter()
            try:
                context_result, context_report = sampler.search_ternary_top_k(
                    steps=steps.filter(pl.col("context_id") == context_id),
                    initial_locations=initial_locations.filter(
                        pl.col("context_id") == context_id
                    ),
                    **search_arguments,
                )
            except ValueError as error:
                failed_contexts.append(context_id)
                print(
                    f"context={context_id} failed={error}",
                    flush=True,
                )
                continue
            results.append(context_result)
            reports.append(context_report)
            elapsed = time.perf_counter() - context_started
            context_timings.append((context_id, elapsed))
            if not args.quiet_contexts:
                print(
                    f"context={context_id} seconds={elapsed:.3f} "
                    f"children={context_report['children_considered']:,}",
                    flush=True,
                )
        result = pl.concat(results) if results else pl.DataFrame()
        report = {
            key: (
                str(sum(int(item[key]) for item in reports))
                if key == "assignment_lattice"
                else sum(item[key] for item in reports)
            )
            for key in reports[0]
        } if reports else {}
        print(
            f"failed contexts: {len(failed_contexts):,} "
            f"{failed_contexts}",
            flush=True,
        )
        sorted_timings = sorted(elapsed for _, elapsed in context_timings)
        if sorted_timings:
            percentile = lambda share: sorted_timings[
                round(share * (len(sorted_timings) - 1))
            ]
            print(
                "context seconds: "
                f"p50={percentile(0.50):.4f} "
                f"p95={percentile(0.95):.4f} "
                f"p99={percentile(0.99):.4f} "
                f"max={sorted_timings[-1]:.4f}",
                flush=True,
            )
            print(
                "slowest successful contexts: "
                f"{sorted(context_timings, key=lambda item: item[1], reverse=True)[:10]}",
                flush=True,
            )
    else:
        result, report = sampler.search_ternary_top_k(
            steps=steps,
            initial_locations=initial_locations,
            **search_arguments,
        )
    search_seconds = time.perf_counter() - started

    print("Grand Genève exact heap search benchmark")
    print(f"preparation seconds: {preparation_seconds:.3f}")
    print(f"index build seconds: {build_seconds:.3f}")
    print(f"contexts: {initial_locations.height:,}")
    print(f"step rows: {steps.height:,}")
    print(f"search seconds: {search_seconds:.3f}")
    print(f"output rows: {result.height:,}")
    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
