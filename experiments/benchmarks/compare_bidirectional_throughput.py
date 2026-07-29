"""Counterbalanced A/B throughput comparison for bounded top-K configurations."""

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
from experiments.harness import (
    ExperimentKind,
    ExperimentManifest,
    RunRecorder,
    linked_verdict,
)
from experiments.top_k_config import ACTIVE_TOP_K_DEFAULTS, apply_top_k_overrides


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
    parser.add_argument(
        "--cycles",
        type=int,
        default=2,
        help="number of counterbalanced four-run blocks (ABBA, then BAAB)",
    )
    parser.add_argument("--exploration-seed", type=int, default=42)
    parser.add_argument("--hypothesis", default="unspecified")
    parser.add_argument(
        "--experiment-manifest",
        type=Path,
        help="immutable A/B configuration, cohort role, and decision gates",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("experiments/runs"),
        help="generated artifact root for manifest-driven runs",
    )
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
        "--max-wall-regression",
        type=float,
        default=0.15,
        help="legacy output-changing wall-time ceiling; quality evidence is still required",
    )
    parser.add_argument(
        "--require-promotion",
        action="store_true",
        help="exit nonzero unless the declared experiment verdict passes",
    )
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def parse_candidate_options(
    values: list[str],
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return apply_top_k_overrides(
        dict(baseline or ACTIVE_TOP_K_DEFAULTS),
        values,
    )


def counterbalanced_sequence(cycles: int) -> list[tuple[int, str]]:
    """Return equal A/B counts with first-order position balanced across blocks."""
    if cycles <= 0:
        raise ValueError("--cycles must be positive")
    patterns = (("A", "B", "B", "A"), ("B", "A", "A", "B"))
    return [
        (block + 1, label)
        for block in range(cycles)
        for label in patterns[block % len(patterns)]
    ]


def paired_block_deltas(
    runs: list[dict[str, Any]],
    key: str,
) -> list[float]:
    """Compare A and B within each counterbalanced block."""
    deltas = []
    for block in sorted({int(run["block"]) for run in runs}):
        block_runs = [run for run in runs if int(run["block"]) == block]
        a = statistics.fmean(
            float(run[key]) for run in block_runs if run["label"] == "A"
        )
        b = statistics.fmean(
            float(run[key]) for run in block_runs if run["label"] == "B"
        )
        deltas.append(b / a - 1.0 if a else float("nan"))
    return deltas


def throughput_verdict(
    *,
    kind: ExperimentKind,
    gates: dict[str, Any],
    relative_delta: dict[str, float],
    output_same: bool,
    counter_drift: list[str],
    quality_status: str = "incomplete",
    quality_reason: str = "no quality evidence required",
) -> dict[str, Any]:
    """Apply only the rubric declared for this experiment kind."""
    failures: list[str] = []
    incomplete: list[str] = []
    if kind is ExperimentKind.PURE_PERF:
        minimum = float(gates["min_rust_improvement"])
        if gates.get("require_same_output", True) and not output_same:
            failures.append("output fingerprints differ")
        if gates.get("require_same_counters", True) and counter_drift:
            failures.append("work counters differ")
        if -relative_delta["rust_seconds"] < minimum:
            failures.append(
                f"Rust improvement {-relative_delta['rust_seconds']:.1%} "
                f"is below {minimum:.1%}"
            )
    elif kind is ExperimentKind.QUALITY_RUNTIME:
        maximum = float(gates["max_wall_regression"])
        if relative_delta["wall_seconds"] > maximum:
            failures.append(
                f"wall regression {relative_delta['wall_seconds']:.1%} "
                f"exceeds {maximum:.1%}"
            )
        if quality_status == "fail":
            failures.append(quality_reason)
        elif quality_status != "pass":
            incomplete.append(quality_reason)
    else:
        raise ValueError(f"{kind.value} cannot be decided by the throughput harness")
    status = "fail" if failures else "incomplete" if incomplete else "pass"
    return {
        "status": status,
        "kind": kind.value,
        "failures": failures,
        "incomplete": incomplete,
    }


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
    "pricing_pair_evaluations",
    "pricing_pair_probes",
    "pricing_pair_expansions",
)


def main() -> None:
    args = parse_args()
    sequence = counterbalanced_sequence(args.cycles)
    manifest = (
        ExperimentManifest.load(args.experiment_manifest)
        if args.experiment_manifest
        else None
    )
    if manifest and manifest.kind is ExperimentKind.QUALITY_ONLY:
        raise ValueError("quality_only manifests must use a quality harness")
    if manifest and args.candidate_option:
        raise ValueError(
            "--candidate-option cannot override an immutable experiment manifest"
        )
    if manifest and args.allow_output_change:
        raise ValueError(
            "--allow-output-change is inferred from the manifest experiment kind"
        )
    if manifest:
        args.hypothesis = manifest.hypothesis
        args.exploration_seed = int(manifest.cohort["selection_seed"])
        if "contexts" in manifest.cohort:
            args.contexts = int(manifest.cohort["contexts"])
        selector = manifest.cohort.get("selector")
        args.calibrated = selector == "calibrated"
        args.all_supported = selector == "all_supported"
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
    selected_context_ids = [int(value) for value in selected["context_id"].to_list()]
    cohort_identity = (
        manifest.verify_cohort(selected_context_ids)
        if manifest
        else None
    )
    recorder = (
        RunRecorder(
            manifest,
            cohort_identity=cohort_identity,
            root=args.run_root,
            metadata={
                "harness": "compare_bidirectional_throughput",
                "contexts": len(selected_context_ids),
                "threads": args.threads,
                "top_k": args.top_k,
                "cycles": args.cycles,
            },
        )
        if manifest and cohort_identity
        else None
    )
    if cohort_identity:
        print(f"Cohort fingerprint: {cohort_identity}")
    if recorder:
        print(f"Run artifact: {recorder.path}")
        recorder.progress("cohort_ready", contexts=len(selected_context_ids))
    steps = steps.join(selected, on="context_id", how="semi")
    initial_locations = initial_locations.join(selected, on="context_id", how="semi")
    search = DestinationPlanSearch(od_costs=od_costs, destination_inputs=destination_inputs)

    baseline = dict(manifest.baseline if manifest else ACTIVE_TOP_K_DEFAULTS)
    candidate = (
        dict(manifest.candidate)
        if manifest
        else parse_candidate_options(args.candidate_option, baseline)
    )
    options_by_label = {"A": baseline, "B": candidate}
    runs: list[dict[str, Any]] = []
    allow_output_change = (
        manifest.kind is ExperimentKind.QUALITY_RUNTIME
        if manifest
        else args.allow_output_change
    )
    for ordinal, (block, label) in enumerate(sequence, start=1):
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
            skip_contexts_without_plan=True,
            collect_profile=True,
        )
        fingerprint = output_fingerprint(output)
        run = {
            "run": ordinal,
            "block": block,
            "label": label,
            "wall_seconds": time.perf_counter() - started,
            "rust_seconds": report["total_search_ns"] / 1e9,
            "factor_map_seconds": report["factor_map_ns"] / 1e9,
            "pricing_seconds": report["pricing_ns"] / 1e9,
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
            f"run={ordinal} block={block} {label} wall={run['wall_seconds']:.3f}s "
            f"rust={run['rust_seconds']:.3f}s factor-map={run['factor_map_seconds']:.3f}s "
            f"pricing={run['pricing_seconds']:.3f}s "
            f"fingerprint={fingerprint}"
        )
        if recorder:
            recorder.progress(
                "measurement",
                run=ordinal,
                block=block,
                label=label,
                wall_seconds=run["wall_seconds"],
                rust_seconds=run["rust_seconds"],
                fingerprint=fingerprint,
            )

    summary: dict[str, dict[str, float]] = {}
    for label in ("A", "B"):
        label_runs = [run for run in runs if run["label"] == label]
        summary[label] = {
            key: statistics.median(float(run[key]) for run in label_runs)
            for key in (
                "wall_seconds",
                "rust_seconds",
                "factor_map_seconds",
                "pricing_seconds",
            )
        }
    delta = {
        key: (
            summary["B"][key] / summary["A"][key] - 1.0
            if summary["A"][key]
            else float("nan")
        )
        for key in summary["A"]
    }
    block_deltas = {
        key: paired_block_deltas(runs, key)
        for key in (
            "wall_seconds",
            "rust_seconds",
            "factor_map_seconds",
            "pricing_seconds",
        )
    }
    paired_delta = {
        key: statistics.median(values)
        for key, values in block_deltas.items()
        if all(value == value for value in values)
    }
    block_ranges = {
        key: [min(values), max(values)]
        for key, values in block_deltas.items()
        if all(value == value for value in values)
    }
    print("\nmedian | wall | Rust aggregate | factor-map aggregate | pricing aggregate")
    for label in ("A", "B"):
        values = summary[label]
        print(
            f"{label} | {values['wall_seconds']:.3f}s | {values['rust_seconds']:.3f}s | "
            f"{values['factor_map_seconds']:.3f}s | {values['pricing_seconds']:.3f}s"
        )
    print("pair work | evaluations | probes | expansions")
    for label in ("A", "B"):
        counters = next(run["counters"] for run in runs if run["label"] == label)
        print(
            f"{label} | {counters['pricing_pair_evaluations']} | "
            f"{counters['pricing_pair_probes']} | "
            f"{counters['pricing_pair_expansions']}"
        )
    counter_reference = runs[0]["counters"]
    counter_drift = [
        f"run {run['run']} {run['label']} {name}={run['counters'][name]} != {expected}"
        for run in runs[1:]
        for name, expected in counter_reference.items()
        if run["counters"][name] != expected
    ]
    fingerprints_by_label = {
        label: sorted({str(run["fingerprint"]) for run in runs if run["label"] == label})
        for label in ("A", "B")
    }
    output_same = (
        len(fingerprints_by_label["A"]) == 1
        and len(fingerprints_by_label["B"]) == 1
        and fingerprints_by_label["A"] == fingerprints_by_label["B"]
    )
    print(
        f"B versus A | {delta['wall_seconds']:+.1%} | {delta['rust_seconds']:+.1%} | "
        f"{delta['factor_map_seconds']:+.1%} | "
        + (
            f"{delta['pricing_seconds']:+.1%}"
            if summary["A"]["pricing_seconds"]
            else f"+{summary['B']['pricing_seconds']:.3f}s"
        )
    )
    if counter_drift:
        print("counter drift (not a pure performance comparison):")
        for value in counter_drift[:10]:
            print(f"  {value}")
    print("paired-block median and observed range")
    for key, value in paired_delta.items():
        lower, upper = block_ranges[key]
        print(f"  {key}: {value:+.1%} [{lower:+.1%}, {upper:+.1%}]")
    kind = (
        manifest.kind
        if manifest
        else ExperimentKind.QUALITY_RUNTIME
        if allow_output_change
        else ExperimentKind.PURE_PERF
    )
    gates = (
        dict(manifest.gates)
        if manifest
        else {
            "max_wall_regression": args.max_wall_regression,
            "min_rust_improvement": args.min_rust_improvement,
            "require_same_output": True,
            "require_same_counters": True,
        }
    )
    quality_status, quality_reason = linked_verdict(
        manifest.evidence_path("quality_artifact")
        if manifest and kind is ExperimentKind.QUALITY_RUNTIME
        else None,
        expected_configs={"A": baseline, "B": candidate}
        if manifest and kind is ExperimentKind.QUALITY_RUNTIME
        else None,
        expected_quality_gates=dict(manifest.gates)
        if manifest and kind is ExperimentKind.QUALITY_RUNTIME
        else None,
    )
    verdict = throughput_verdict(
        kind=kind,
        gates=gates,
        relative_delta=paired_delta,
        output_same=output_same,
        counter_drift=counter_drift,
        quality_status=quality_status,
        quality_reason=quality_reason,
    )
    print(
        "verdict | "
        + verdict["status"].upper()
        + f" | kind={kind.value} | hypothesis={args.hypothesis}"
    )
    for reason in verdict["failures"] + verdict["incomplete"]:
        print(f"  {reason}")
    payload = {
        "hypothesis": args.hypothesis,
        "kind": kind.value,
        "calibrated": args.calibrated,
        "cohort_fingerprint": cohort_identity,
        "resolved_configs": {"A": baseline, "B": candidate},
        "runs": runs,
        "summary": summary,
        "relative_delta": delta,
        "paired_block_delta": block_deltas,
        "paired_block_median": paired_delta,
        "paired_block_range": block_ranges,
        "fingerprints_by_label": fingerprints_by_label,
        "output_same": output_same,
        "counter_drift": counter_drift,
        "verdict": verdict,
    }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.json_output}")
    if recorder:
        recorder.finalize(payload, status=verdict["status"])
    if args.require_promotion and verdict["status"] != "pass":
        raise SystemExit("candidate did not clear the declared experiment gate")


if __name__ == "__main__":
    main()
