"""Evaluate observable routing signals for iterative exact path pricing.

The script reads only cached exact top-K certificates, then runs the active
bounded search with zero, one, and two pricing passes. It reports how much of
the oracle-measured gain a router could capture using signals available without
the oracle, especially changes produced by the first pricing pass.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from mobility_destination_sequence_sampler import DestinationPlanSearch

from experiments.analysis.compare_bidirectional_top_k_grand_geneve import (
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
from experiments.oracle_cache import OracleCache, oracle_input_fingerprint
from experiments.top_k_config import ACTIVE_TOP_K_DEFAULTS


@dataclass
class Case:
    context_id: int
    layers: int
    baseline_mass: float
    one_pass_mass: float
    two_pass_mass: float
    one_pass_new_plans: int
    one_pass_top_delta: float
    one_pass_kth_delta: float
    baseline_score_span: float
    completed_plans: int
    baseline_ms: float
    one_pass_ms: float
    two_pass_ms: float
    active_mass: float
    active_ms: float
    pair_masses: dict[int, float]
    pair_ms: dict[int, float]
    pair_evaluations: dict[int, int]

    @property
    def first_gain(self) -> float:
        return self.one_pass_mass - self.baseline_mass

    @property
    def second_gain(self) -> float:
        return self.two_pass_mass - self.one_pass_mass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group-day-trips-folder",
        type=Path,
        default=DEFAULT_GROUP_DAY_TRIPS_FOLDER,
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-states", type=int, default=500_000)
    parser.add_argument("--frontier-width", type=int, default=32)
    parser.add_argument("--pricing-min-layers", type=int, default=6)
    parser.add_argument("--pricing-seed-limit", type=int, default=10)
    parser.add_argument("--pricing-column-limit", type=int, default=4)
    parser.add_argument(
        "--oracle-cache-fingerprint",
        help="reuse completed exact certificates from this explicit fingerprint; never reuses cached failures",
    )
    parser.add_argument(
        "--pair-limit",
        type=int,
        action="append",
        default=[],
        help="also compare the active adaptive policy with this interacting-pair candidate limit",
    )
    return parser.parse_args()


def path_set(table: pl.DataFrame) -> set[tuple[int, ...]]:
    return {zones for zones, _ in ranked_plans(table)}


def run_search(
    search: DestinationPlanSearch,
    context_steps: pl.DataFrame,
    context_initial: pl.DataFrame,
    args: argparse.Namespace,
    pricing_passes: int,
    *,
    pair_limit: int = 0,
    pricing_next_pass_min_new: int = 0,
) -> tuple[pl.DataFrame, dict[str, int]]:
    options = {
        **ACTIVE_TOP_K_DEFAULTS,
        "frontier_width": args.frontier_width,
        "pricing_passes": pricing_passes,
        "pricing_seed_limit": args.pricing_seed_limit,
        "pricing_column_limit": args.pricing_column_limit,
        "pricing_pair_candidate_limit": pair_limit,
        "pricing_pair_deep_candidate_limit": 0,
        "pricing_next_pass_min_new": pricing_next_pass_min_new,
        "pricing_min_layers": args.pricing_min_layers,
    }
    return search.top_k(
        steps=context_steps,
        initial_locations=context_initial,
        logit_scale=LOGIT_SCALE,
        update_plan_timings=True,
        use_shadow_prices=True,
        exploration_seed=42,
        **options,
        top_k=args.top_k,
        n_threads=1,
        skip_infeasible=True,
        collect_profile=True,
    )


def score_at_rank(plans: list[tuple[tuple[int, ...], float]], rank: int) -> float:
    return plans[min(rank, len(plans)) - 1][1] if plans else float("-inf")


def router_row(
    label: str,
    cases: list[Case],
    selected: list[Case],
    gain_name: str,
) -> str:
    total_gain = sum(max(getattr(case, gain_name), 0.0) for case in cases)
    captured = sum(max(getattr(case, gain_name), 0.0) for case in selected)
    return (
        f"{label:30s} | {len(selected) / max(len(cases), 1):6.1%} | "
        f"{captured / total_gain if total_gain else 1.0:6.1%} | "
        f"{sum(getattr(case, gain_name) for case in selected):+.3f}"
    )


def print_router_study(cases: list[Case]) -> None:
    print(
        f"certified deep contexts={len(cases)}; mean Mass@10 "
        f"{sum(case.baseline_mass for case in cases) / len(cases):.3f}"
        f"->{sum(case.one_pass_mass for case in cases) / len(cases):.3f}"
        f"->{sum(case.two_pass_mass for case in cases) / len(cases):.3f}"
    )
    print(
        "mean bounded ms "
        f"{sum(case.baseline_ms for case in cases) / len(cases):.2f}"
        f"->{sum(case.one_pass_ms for case in cases) / len(cases):.2f}"
        f"->{sum(case.two_pass_ms for case in cases) / len(cases):.2f}"
    )

    print("\nFirst-pass routing from baseline signals")
    print("rule                           | routed | captured positive gain | signed gain")
    first_rules = [
        ("all depth>=6", cases),
        ("depth>=7", [case for case in cases if case.layers >= 7]),
        ("depth>=8", [case for case in cases if case.layers >= 8]),
        (
            "completed plans<100",
            [case for case in cases if case.completed_plans < 100],
        ),
        (
            "returned score span<1",
            [case for case in cases if case.baseline_score_span < 1.0],
        ),
        (
            "returned score span<2",
            [case for case in cases if case.baseline_score_span < 2.0],
        ),
    ]
    for label, selected in first_rules:
        print(router_row(label, cases, selected, "first_gain"))

    print("\nSecond-pass routing from first-pass signals")
    print("rule                           | routed | captured positive gain | signed gain")
    second_rules = [
        ("all first-pass contexts", cases),
        (
            "new surviving plans>=1",
            [case for case in cases if case.one_pass_new_plans >= 1],
        ),
        (
            "new surviving plans>=3",
            [case for case in cases if case.one_pass_new_plans >= 3],
        ),
        (
            "new surviving plans>=5",
            [case for case in cases if case.one_pass_new_plans >= 5],
        ),
        (
            "top score improved",
            [case for case in cases if case.one_pass_top_delta > 1e-12],
        ),
        (
            "kth score improved",
            [case for case in cases if case.one_pass_kth_delta > 1e-12],
        ),
        (
            "kth delta>0.1",
            [case for case in cases if case.one_pass_kth_delta > 0.1],
        ),
        (
            "kth delta>0.5",
            [case for case in cases if case.one_pass_kth_delta > 0.5],
        ),
    ]
    for label, selected in second_rules:
        print(router_row(label, cases, selected, "second_gain"))

    print("\nLargest second-pass gains")
    for case in sorted(cases, key=lambda item: (-item.second_gain, item.context_id))[:12]:
        print(
            f"context={case.context_id:5d} layers={case.layers:2d} "
            f"mass={case.baseline_mass:.3f}->{case.one_pass_mass:.3f}"
            f"->{case.two_pass_mass:.3f} new={case.one_pass_new_plans:2d} "
            f"top-delta={case.one_pass_top_delta:.3f} "
            f"kth-delta={case.one_pass_kth_delta:.3f}"
        )


def print_pair_study(cases: list[Case], pair_limits: list[int]) -> None:
    print("\nInteracting-pair pricing")
    print("pair limit | n | mean mass | gain | wins/losses | zero | mean ms | pair eval/context")
    baseline_mass = sum(case.active_mass for case in cases) / len(cases)
    baseline_ms = sum(case.active_ms for case in cases) / len(cases)
    baseline_zero = sum(case.active_mass <= 0.0 for case in cases)
    print(
        f"{0:10d} | {len(cases):2d} | {baseline_mass:.3f} | {0.0:+.3f} | "
        f"{0:2d}/{0:2d} | {baseline_zero:4d} | {baseline_ms:7.2f} | {0:17d}"
    )
    for limit in pair_limits:
        masses = [case.pair_masses[limit] for case in cases]
        gains = [mass - case.active_mass for case, mass in zip(cases, masses, strict=True)]
        print(
            f"{limit:10d} | {len(cases):2d} | {sum(masses) / len(cases):.3f} | "
            f"{sum(gains) / len(cases):+.3f} | "
            f"{sum(gain > 1e-12 for gain in gains):2d}/"
            f"{sum(gain < -1e-12 for gain in gains):2d} | "
            f"{sum(mass <= 0.0 for mass in masses):4d} | "
            f"{sum(case.pair_ms[limit] for case in cases) / len(cases):7.2f} | "
            f"{sum(case.pair_evaluations[limit] for case in cases) // len(cases):17d}"
        )
    if 4 in pair_limits and 8 in pair_limits:
        hybrid_masses = [
            case.pair_masses[8] if case.layers >= 9 else case.pair_masses[4]
            for case in cases
        ]
        hybrid_gains = [
            mass - case.active_mass
            for case, mass in zip(cases, hybrid_masses, strict=True)
        ]
        hybrid_ms = [
            case.pair_ms[8] if case.layers >= 9 else case.pair_ms[4]
            for case in cases
        ]
        hybrid_evaluations = [
            case.pair_evaluations[8]
            if case.layers >= 9
            else case.pair_evaluations[4]
            for case in cases
        ]
        print(
            f"{'4; 8@9+':>10s} | {len(cases):2d} | "
            f"{sum(hybrid_masses) / len(cases):.3f} | "
            f"{sum(hybrid_gains) / len(cases):+.3f} | "
            f"{sum(gain > 1e-12 for gain in hybrid_gains):2d}/"
            f"{sum(gain < -1e-12 for gain in hybrid_gains):2d} | "
            f"{sum(mass <= 0.0 for mass in hybrid_masses):4d} | "
            f"{sum(hybrid_ms) / len(cases):7.2f} | "
            f"{sum(hybrid_evaluations) // len(cases):17d}"
        )
    best_limit = max(
        pair_limits,
        key=lambda limit: sum(case.pair_masses[limit] for case in cases),
    )
    print(f"\nLargest gains at pair limit {best_limit}")
    for case in sorted(
        cases,
        key=lambda item: (
            -(item.pair_masses[best_limit] - item.active_mass),
            item.context_id,
        ),
    )[:12]:
        gain = case.pair_masses[best_limit] - case.active_mass
        print(
            f"context={case.context_id:5d} layers={case.layers:2d} "
            f"mass={case.active_mass:.3f}->{case.pair_masses[best_limit]:.3f} "
            f"gain={gain:+.3f}"
        )


def main() -> None:
    args = parse_args()
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
    fingerprint = args.oracle_cache_fingerprint or oracle_input_fingerprint(
        files,
        logit_scale=LOGIT_SCALE,
        update_plan_timings=True,
        use_shadow_prices=True,
    )
    cache = OracleCache(fingerprint, args.top_k, args.max_states)
    context_ids = cache.cached_context_ids()
    search = DestinationPlanSearch(
        od_costs=od_costs,
        destination_inputs=destination_inputs,
    )
    cases = []
    for context_id in context_ids:
        context_steps = steps.filter(pl.col("context_id") == context_id).sort("layer")
        if context_steps.height < args.pricing_min_layers:
            continue
        cached = cache.load_cached(context_id)
        if cached is None:
            continue
        oracle_table, _ = cached
        context_initial = initial_locations.filter(pl.col("context_id") == context_id)
        baseline, baseline_report = run_search(
            search, context_steps, context_initial, args, 0
        )
        one_pass, one_report = run_search(
            search, context_steps, context_initial, args, 1
        )
        two_pass, two_report = run_search(
            search, context_steps, context_initial, args, 2
        )
        oracle = ranked_plans(oracle_table)
        baseline_plans = ranked_plans(baseline)
        one_plans = ranked_plans(one_pass)
        two_plans = ranked_plans(two_pass)
        if not baseline_plans or not one_plans or not two_plans:
            continue
        active_mass = retained_probability_mass(oracle, path_set(two_pass))
        active_ms = float(two_report["total_search_ns"]) / 1e6
        pair_masses = {}
        pair_ms = {}
        pair_evaluations = {}
        if args.pair_limit:
            active, active_report = run_search(
                search,
                context_steps,
                context_initial,
                args,
                2,
                pricing_next_pass_min_new=ACTIVE_TOP_K_DEFAULTS[
                    "pricing_next_pass_min_new"
                ],
            )
            active_mass = retained_probability_mass(oracle, path_set(active))
            active_ms = float(active_report["total_search_ns"]) / 1e6
            for pair_limit in args.pair_limit:
                paired, paired_report = run_search(
                    search,
                    context_steps,
                    context_initial,
                    args,
                    2,
                    pair_limit=pair_limit,
                    pricing_next_pass_min_new=ACTIVE_TOP_K_DEFAULTS[
                        "pricing_next_pass_min_new"
                    ],
                )
                pair_masses[pair_limit] = retained_probability_mass(
                    oracle, path_set(paired)
                )
                pair_ms[pair_limit] = float(paired_report["total_search_ns"]) / 1e6
                pair_evaluations[pair_limit] = int(
                    paired_report["pricing_pair_evaluations"]
                )
        baseline_paths = path_set(baseline)
        one_paths = path_set(one_pass)
        cases.append(
            Case(
                context_id=context_id,
                layers=context_steps.height,
                baseline_mass=retained_probability_mass(oracle, baseline_paths),
                one_pass_mass=retained_probability_mass(oracle, one_paths),
                two_pass_mass=retained_probability_mass(oracle, path_set(two_pass)),
                one_pass_new_plans=len(one_paths - baseline_paths),
                one_pass_top_delta=one_plans[0][1] - baseline_plans[0][1],
                one_pass_kth_delta=score_at_rank(one_plans, args.top_k)
                - score_at_rank(baseline_plans, args.top_k),
                baseline_score_span=baseline_plans[0][1]
                - score_at_rank(baseline_plans, args.top_k),
                completed_plans=int(baseline_report["complete_plan_candidates"]),
                baseline_ms=float(baseline_report["total_search_ns"]) / 1e6,
                one_pass_ms=float(one_report["total_search_ns"]) / 1e6,
                two_pass_ms=float(two_report["total_search_ns"]) / 1e6,
                active_mass=active_mass,
                active_ms=active_ms,
                pair_masses=pair_masses,
                pair_ms=pair_ms,
                pair_evaluations=pair_evaluations,
            )
        )
    if not cases:
        raise RuntimeError(f"no cached deep certificates under {cache.path}")
    print_router_study(cases)
    if args.pair_limit:
        print_pair_study(cases, list(dict.fromkeys(args.pair_limit)))


if __name__ == "__main__":
    main()
