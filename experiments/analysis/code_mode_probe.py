"""Run bounded and exact searches locally, returning only a compact JSON decision report.

This is a code-mode-style diagnostic for an agent: Polars inputs, result tables,
and complete Rust reports stay inside this process. Stdout contains only the
explicitly selected summary fields, so an agent can inspect several contexts
without carrying the raw search outputs in its conversation context.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl

from mobility_destination_sequence_sampler import DestinationPlanSearch

from experiments.analysis.compare_bidirectional_top_k_grand_geneve import (
    oracle_failure_kind,
    ranked_plans,
    retained_probability_mass,
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
    parser.add_argument(
        "--context-id",
        type=int,
        action="append",
        required=True,
        help="context to compare; repeat to batch multiple contexts in one process",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--oracle-depth",
        type=int,
        default=10,
        help="exact support retained locally for each context; must be at least --top-k",
    )
    parser.add_argument("--max-states", type=int, default=500_000)
    parser.add_argument(
        "--no-bounded-incumbent",
        action="store_true",
        help="disable exact rescoring of active bounded plans as oracle lower bounds",
    )
    parser.add_argument("--exploration-seed", type=int, default=42)
    parser.add_argument(
        "--json-output",
        type=Path,
        help="optional copy of the same compact report; stdout remains JSON only",
    )
    return parser.parse_args()


def compare_plans(
    oracle: list[tuple[tuple[int, ...], float]],
    bounded: list[tuple[tuple[int, ...], float]],
    top_k: int,
) -> dict[str, float | int]:
    """Summarize two local plan tables without returning either table."""
    oracle_top_k = oracle[:top_k]
    returned_zones = {zones for zones, _ in bounded}
    recovered = sum(zones in returned_zones for zones, _ in oracle_top_k)
    return {
        "exact_plans": len(oracle),
        "bounded_plans": len(bounded),
        "recovered": recovered,
        "recall_at_k": recovered / len(oracle_top_k) if oracle_top_k else 0.0,
        "mass_at_k": retained_probability_mass(oracle_top_k, returned_zones),
    }


def compact_report(
    cases: list[dict[str, Any]],
    *,
    top_k: int,
    oracle_depth: int,
    max_states: int,
    use_bounded_incumbent: bool,
) -> dict[str, Any]:
    """Create the only object a caller needs to receive from the batch."""
    compared = [case for case in cases if case["outcome"] == "compared"]
    return {
        "schema_version": 1,
        "mode": "local-code-mode-probe",
        "parameters": {
            "top_k": top_k,
            "oracle_depth": oracle_depth,
            "max_states": max_states,
            "use_bounded_incumbent": use_bounded_incumbent,
        },
        "summary": {
            "requested_contexts": len(cases),
            "compared_contexts": len(compared),
            "oracle_unproven_contexts": sum(
                case["outcome"] == "oracle_unproven" for case in cases
            ),
            "bounded_failed_contexts": sum(
                case["outcome"] == "bounded_failed" for case in cases
            ),
            "mean_recall_at_k": (
                sum(float(case["recall_at_k"]) for case in compared) / len(compared)
                if compared
                else None
            ),
            "mean_mass_at_k": (
                sum(float(case["mass_at_k"]) for case in compared) / len(compared)
                if compared
                else None
            ),
        },
        "cases": cases,
    }


def main() -> None:
    args = parse_args()
    if args.top_k <= 0 or args.oracle_depth < args.top_k or args.max_states <= 0:
        raise ValueError("top-k, oracle-depth, and max-states are inconsistent")

    files = resolve_snapshot_files(args.group_day_trips_folder)
    od_costs = prepare_od_costs(files["transport_costs"], files["demand_groups"])
    destination_inputs = prepare_destination_inputs(
        files["destination_saturation"], files["demand_groups"]
    )
    steps, initial_locations, _ = prepare_complete_contexts(
        activity_sequences_path=files["activity_sequences"],
        survey_plan_steps_path=files["survey_plan_steps"],
        demand_groups_path=files["demand_groups"],
        activity_dur_path=files["activity_dur"],
    )
    search = DestinationPlanSearch(od_costs=od_costs, destination_inputs=destination_inputs)

    cases: list[dict[str, Any]] = []
    for context_id in dict.fromkeys(args.context_id):
        context_steps = steps.filter(pl.col("context_id") == context_id)
        context_initial = initial_locations.filter(pl.col("context_id") == context_id)
        if context_steps.height == 0 or context_initial.height != 1:
            cases.append({"context_id": context_id, "outcome": "input_missing"})
            continue
        try:
            oracle_table, oracle_report = search.exact_top_k(
                steps=context_steps,
                initial_locations=context_initial,
                logit_scale=LOGIT_SCALE,
                update_plan_timings=True,
                use_shadow_prices=True,
                top_k=args.oracle_depth,
                max_states=args.max_states,
                n_threads=1,
                skip_infeasible=False,
                use_bounded_incumbent=not args.no_bounded_incumbent,
            )
        except ValueError as error:
            cases.append(
                {
                    "context_id": context_id,
                    "layers": context_steps.height,
                    "outcome": "oracle_unproven",
                    "reason": oracle_failure_kind(error),
                }
            )
            continue
        try:
            bounded_table, bounded_report = search.top_k(
                steps=context_steps,
                initial_locations=context_initial,
                logit_scale=LOGIT_SCALE,
                update_plan_timings=True,
                use_shadow_prices=True,
                exploration_seed=args.exploration_seed,
                **ACTIVE_TOP_K_DEFAULTS,
                top_k=args.top_k,
                n_threads=1,
                skip_infeasible=False,
                collect_profile=True,
            )
        except ValueError:
            cases.append(
                {
                    "context_id": context_id,
                    "layers": context_steps.height,
                    "outcome": "bounded_failed",
                    "oracle_states_pushed": int(oracle_report["states_pushed"]),
                    "oracle_incumbent_plans_seeded": int(
                        oracle_report["incumbent_plans_seeded"]
                    ),
                    "oracle_children_pruned_by_incumbent": int(
                        oracle_report["children_pruned_by_incumbent"]
                    ),
                }
            )
            continue
        cases.append(
            {
                "context_id": context_id,
                "layers": context_steps.height,
                "outcome": "compared",
                "oracle_states_pushed": int(oracle_report["states_pushed"]),
                "oracle_incumbent_plans_seeded": int(
                    oracle_report["incumbent_plans_seeded"]
                ),
                "oracle_children_pruned_by_incumbent": int(
                    oracle_report["children_pruned_by_incumbent"]
                ),
                "bounded_search_ms": round(bounded_report["total_search_ns"] / 1e6, 3),
                **compare_plans(ranked_plans(oracle_table), ranked_plans(bounded_table), args.top_k),
            }
        )

    report = compact_report(
        cases,
        top_k=args.top_k,
        oracle_depth=args.oracle_depth,
        max_states=args.max_states,
        use_bounded_incumbent=not args.no_bounded_incumbent,
    )
    encoded = json.dumps(report, separators=(",", ":"), sort_keys=True)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
