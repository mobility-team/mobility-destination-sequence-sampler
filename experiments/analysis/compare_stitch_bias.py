"""Compare balanced and asymmetric stitch boundaries against the exact oracle."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from mobility_destination_sequence_sampler import DestinationPlanSearch

from experiments.analysis.compare_bidirectional_top_k_grand_geneve import (
    eligible_context_ids,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group-day-trips-folder",
        type=Path,
        default=DEFAULT_GROUP_DAY_TRIPS_FOLDER,
    )
    parser.add_argument("--contexts", type=int, default=50)
    parser.add_argument("--candidate-contexts", type=int, default=300)
    parser.add_argument("--max-layers", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--oracle-depth", type=int, default=100)
    parser.add_argument("--max-states", type=int, default=2_000_000)
    parser.add_argument("--left-bias", type=int, default=0)
    parser.add_argument("--right-bias", type=int, default=1)
    parser.add_argument("--frontier-width", type=int, default=32)
    parser.add_argument("--proposal-limit-per-source", type=int, default=16)
    parser.add_argument("--continuation-state-limit", type=int, default=1)
    parser.add_argument("--continuation-proposal-limit", type=int, default=1)
    parser.add_argument("--exploration-seed", type=int, default=42)
    return parser.parse_args()


def run_bounded(
    search: DestinationPlanSearch,
    steps: object,
    initial_locations: object,
    args: argparse.Namespace,
    stitch_bias: int,
) -> list[tuple[tuple[int, ...], float]]:
    table, _ = search.top_k(
        steps=steps,
        initial_locations=initial_locations,
        logit_scale=LOGIT_SCALE,
        update_plan_timings=True,
        use_shadow_prices=True,
        exploration_seed=args.exploration_seed,
        frontier_width=args.frontier_width,
        proposal_limit_per_source=args.proposal_limit_per_source,
        stitch_bias=stitch_bias,
        continuation_state_limit=args.continuation_state_limit,
        continuation_proposal_limit=args.continuation_proposal_limit,
        top_k=args.top_k,
        n_threads=1,
        skip_infeasible=False,
    )
    return ranked_plans(table)


def print_examples(label: str, rows: list[dict[str, object]]) -> None:
    print(f"\n{label}")
    for row in rows:
        print(
            f"context={row['context_id']} layers={row['layers']} "
            f"delta={row['delta']:+.4f} "
            f"left={row['left_mass']:.4f} right={row['right_mass']:.4f}"
        )
        print(f"  steps: {row['structure']}")
        for direction in ("gained", "lost"):
            plans = row[direction]
            if plans:
                print(f"  {direction}: {plans}")


def main() -> None:
    args = parse_args()
    files = resolve_snapshot_files(args.group_day_trips_folder)
    print("Preparing cached Grand Geneve inputs (read-only)...")
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
    selection_args = SimpleNamespace(
        contexts=args.contexts,
        candidate_contexts=args.candidate_contexts,
        max_layers=args.max_layers,
        trace_context=None,
        exploration_seed=args.exploration_seed,
    )
    search = DestinationPlanSearch(
        od_costs=od_costs,
        destination_inputs=destination_inputs,
    )
    rows: list[dict[str, object]] = []
    skipped = 0
    for context_id in eligible_context_ids(steps, selection_args):
        context_steps = steps.filter(steps["context_id"] == context_id)
        context_initial = initial_locations.filter(
            initial_locations["context_id"] == context_id
        )
        try:
            oracle_table, _ = search.exact_top_k(
                steps=context_steps,
                initial_locations=context_initial,
                logit_scale=LOGIT_SCALE,
                update_plan_timings=True,
                use_shadow_prices=True,
                top_k=args.oracle_depth,
                max_states=args.max_states,
                n_threads=1,
                skip_infeasible=False,
            )
            left = run_bounded(search, context_steps, context_initial, args, args.left_bias)
            right = run_bounded(search, context_steps, context_initial, args, args.right_bias)
        except ValueError:
            skipped += 1
            continue
        oracle = ranked_plans(oracle_table)
        oracle_top_k = oracle[: args.top_k]
        left_zones = {zones for zones, _ in left}
        right_zones = {zones for zones, _ in right}
        left_mass = retained_probability_mass(oracle_top_k, left_zones)
        right_mass = retained_probability_mass(oracle_top_k, right_zones)
        rank_by_zones = {zones: rank for rank, (zones, _) in enumerate(oracle_top_k, start=1)}
        rows.append(
            {
                "context_id": context_id,
                "layers": context_steps.height,
                "left_mass": left_mass,
                "right_mass": right_mass,
                "delta": right_mass - left_mass,
                "structure": [
                    (
                        step["activity_id"],
                        step["anchor_id"],
                        step["fixed_destination"],
                    )
                    for step in context_steps.sort("layer").to_dicts()
                ],
                "gained": [
                    (rank_by_zones[zones], zones)
                    for zones in right_zones - left_zones
                    if zones in rank_by_zones
                ],
                "lost": [
                    (rank_by_zones[zones], zones)
                    for zones in left_zones - right_zones
                    if zones in rank_by_zones
                ],
            }
        )
        if len(rows) == args.contexts:
            break
    if not rows:
        raise RuntimeError("no context completed the exact oracle")
    mean = lambda key: sum(float(row[key]) for row in rows) / len(rows)
    print(
        f"\nproven={len(rows)} skipped={skipped} "
        f"left-bias={args.left_bias} mass={mean('left_mass'):.6f} "
        f"right-bias={args.right_bias} mass={mean('right_mass'):.6f} "
        f"mean-delta={mean('delta'):+.6f}"
    )
    print_examples("largest gains", sorted(rows, key=lambda row: float(row["delta"]), reverse=True)[:8])
    print_examples("largest losses", sorted(rows, key=lambda row: float(row["delta"]))[:8])


if __name__ == "__main__":
    main()
