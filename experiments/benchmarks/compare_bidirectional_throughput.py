"""Interleaved A/B/A throughput comparison for bounded top-K configurations."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import polars as pl

from mobility_destination_sequence_sampler import DestinationPlanSearch

from experiments.benchmarks.perf_bidirectional_grand_geneve import (
    calibrated_contexts,
    eligible_contexts,
)
from experiments.benchmarks.perf_grand_geneve_cache import (
    DEFAULT_GROUP_DAY_TRIPS_FOLDER,
    LOGIT_SCALE,
    prepare_complete_contexts,
    prepare_destination_inputs,
    prepare_od_costs,
    resolve_snapshot_files,
)
from experiments.top_k_config import ACTIVE_TOP_K_DEFAULTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-day-trips-folder", type=Path, default=DEFAULT_GROUP_DAY_TRIPS_FOLDER)
    parser.add_argument("--contexts", type=int, default=1_000)
    parser.add_argument("--all-supported", action="store_true")
    parser.add_argument(
        "--calibrated",
        action="store_true",
        help="use a fixed depth/anchor-stratified cohort instead of the smoke sample",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--cycles", type=int, default=2, help="number of B then A pairs after the first A")
    parser.add_argument("--exploration-seed", type=int, default=42)
    parser.add_argument("--hypothesis", default="unspecified")
    parser.add_argument(
        "--candidate-option",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="override one active top-K option for B; repeat as needed",
    )
    parser.add_argument(
        "--allow-output-change",
        action="store_true",
        help="permit different output fingerprints (for quality/runtime parameter trade-offs)",
    )
    parser.add_argument(
        "--min-rust-improvement",
        type=float,
        default=0.03,
        help="minimum median aggregate-Rust improvement required for promotion",
    )
    parser.add_argument(
        "--require-promotion",
        action="store_true",
        help="exit nonzero unless fingerprints/counters agree and B clears the improvement gate",
    )
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def parse_candidate_options(values: list[str]) -> dict[str, Any]:
    options = dict(ACTIVE_TOP_K_DEFAULTS)
    for value in values:
        name, separator, raw = value.partition("=")
        if not separator or name not in options:
            raise ValueError(f"--candidate-option must name an active option: {value}")
        current = options[name]
        if isinstance(current, int):
            options[name] = int(raw)
        elif isinstance(current, float):
            options[name] = float(raw)
        else:
            options[name] = raw
    return options


def output_fingerprint(table: pl.DataFrame) -> str:
    """Stable identity check for performance-only comparisons."""
    digest = hashlib.sha256()
    columns = [
        "context_id",
        "draw_id",
        "layer",
        "origin",
        "destination",
        "local_log_weight",
        "total_log_weight",
    ]
    for row in table.sort(["context_id", "draw_id", "layer"]).select(columns).iter_rows():
        digest.update(repr(row).encode())
        digest.update(b"\n")
    return digest.hexdigest()[:16]


COUNTERS = (
    "forward_proposals_evaluated",
    "backward_proposals_evaluated",
    "factor_map_previous_destination_scans",
    "factor_map_current_destination_scans",
    "factor_map_next_destination_scans",
    "reverse_prefix_partial_calls",
    "local_score_cache_hits",
    "local_score_cache_builds",
    "complete_plan_candidates",
)


def main() -> None:
    args = parse_args()
    if args.cycles <= 0:
        raise ValueError("--cycles must be positive")
    files = resolve_snapshot_files(args.group_day_trips_folder)
    print("Preparing cached Grand Geneve inputs once (read-only)...")
    od_costs = prepare_od_costs(files["transport_costs"], files["demand_groups"])
    destination_inputs = prepare_destination_inputs(files["destination_saturation"], files["demand_groups"])
    steps, initial_locations, _ = prepare_complete_contexts(
        activity_sequences_path=files["activity_sequences"],
        survey_plan_steps_path=files["survey_plan_steps"],
        demand_groups_path=files["demand_groups"],
        activity_dur_path=files["activity_dur"],
    )
    selected = (
        calibrated_contexts(steps, args.contexts, args.exploration_seed)
        if args.calibrated
        else eligible_contexts(steps, args.contexts, args.exploration_seed, args.all_supported)
    )
    steps = steps.join(selected, on="context_id", how="semi")
    initial_locations = initial_locations.join(selected, on="context_id", how="semi")
    search = DestinationPlanSearch(od_costs=od_costs, destination_inputs=destination_inputs)

    baseline = dict(ACTIVE_TOP_K_DEFAULTS)
    candidate = parse_candidate_options(args.candidate_option)
    sequence = ["A"] + [label for _ in range(args.cycles) for label in ("B", "A")]
    options_by_label = {"A": baseline, "B": candidate}
    runs: list[dict[str, Any]] = []
    reference_fingerprint: str | None = None
    for ordinal, label in enumerate(sequence, start=1):
        started = time.perf_counter()
        output, report = search.top_k(
            steps=steps,
            initial_locations=initial_locations,
            logit_scale=LOGIT_SCALE,
            update_plan_timings=True,
            use_shadow_prices=True,
            exploration_seed=args.exploration_seed,
            **options_by_label[label],
            top_k=args.top_k,
            n_threads=args.threads,
            skip_infeasible=True,
            collect_profile=True,
        )
        fingerprint = output_fingerprint(output)
        if reference_fingerprint is None:
            reference_fingerprint = fingerprint
        elif not args.allow_output_change and fingerprint != reference_fingerprint:
            raise RuntimeError(
                f"output fingerprint changed on run {ordinal} ({label}): "
                f"{fingerprint} != {reference_fingerprint}"
            )
        run = {
            "run": ordinal,
            "label": label,
            "wall_seconds": time.perf_counter() - started,
            "rust_seconds": report["total_search_ns"] / 1e9,
            "factor_map_seconds": report["factor_map_ns"] / 1e9,
            "fingerprint": fingerprint,
            "factor_map_scans": sum(
                report[f"factor_map_{name}_destination_scans"]
                for name in ("previous", "current", "next")
            ),
            "local_cache_hits": report["local_score_cache_hits"],
            "local_cache_builds": report["local_score_cache_builds"],
            "counters": {name: int(report[name]) for name in COUNTERS},
        }
        runs.append(run)
        print(
            f"run={ordinal} {label} wall={run['wall_seconds']:.3f}s "
            f"rust={run['rust_seconds']:.3f}s factor-map={run['factor_map_seconds']:.3f}s "
            f"fingerprint={fingerprint}"
        )

    summary: dict[str, dict[str, float]] = {}
    for label in ("A", "B"):
        label_runs = [run for run in runs if run["label"] == label]
        summary[label] = {
            key: statistics.median(float(run[key]) for run in label_runs)
            for key in ("wall_seconds", "rust_seconds", "factor_map_seconds")
        }
    delta = {
        key: summary["B"][key] / summary["A"][key] - 1.0
        for key in summary["A"]
    }
    print("\nmedian | wall | Rust aggregate | factor-map aggregate")
    for label in ("A", "B"):
        values = summary[label]
        print(
            f"{label} | {values['wall_seconds']:.3f}s | {values['rust_seconds']:.3f}s | "
            f"{values['factor_map_seconds']:.3f}s"
        )
    counter_reference = runs[0]["counters"]
    counter_drift = [
        f"run {run['run']} {run['label']} {name}={run['counters'][name]} != {expected}"
        for run in runs[1:]
        for name, expected in counter_reference.items()
        if run["counters"][name] != expected
    ]
    promoted = (
        not args.allow_output_change
        and not counter_drift
        and -delta["rust_seconds"] >= args.min_rust_improvement
    )
    print(
        f"B versus A | {delta['wall_seconds']:+.1%} | {delta['rust_seconds']:+.1%} | "
        f"{delta['factor_map_seconds']:+.1%}"
    )
    if counter_drift:
        print("counter drift (not a pure performance comparison):")
        for value in counter_drift[:10]:
            print(f"  {value}")
    print(
        "verdict | "
        + ("PROMOTE" if promoted else "do not promote")
        + f" | hypothesis={args.hypothesis} | required Rust improvement={args.min_rust_improvement:.1%}"
    )
    payload = {
        "hypothesis": args.hypothesis,
        "calibrated": args.calibrated,
        "runs": runs,
        "summary": summary,
        "relative_delta": delta,
        "counter_drift": counter_drift,
        "promoted": promoted,
    }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.json_output}")
    if args.require_promotion and not promoted:
        raise SystemExit("candidate did not clear the performance promotion gate")


if __name__ == "__main__":
    main()
