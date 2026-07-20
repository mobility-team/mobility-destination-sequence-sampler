"""Measure the F-to-B seam refresh against exact plans on named contexts."""

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
    parser.add_argument("--context-id", type=int, action="append", dest="context_ids")
    parser.add_argument(
        "--contexts",
        type=int,
        help="Select this many oracle-proven contexts instead of named contexts.",
    )
    parser.add_argument("--candidate-contexts", type=int, default=300)
    parser.add_argument("--max-layers", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--oracle-depth", type=int, default=100)
    parser.add_argument("--max-states", type=int, default=2_000_000)
    parser.add_argument("--frontier-width", type=int, default=32)
    parser.add_argument("--proposal-limit-per-source", type=int, default=16)
    parser.add_argument("--continuation-state-limit", type=int, default=1)
    parser.add_argument("--continuation-proposal-limit", type=int, default=1)
    parser.add_argument("--seam-refresh-per-prefix", type=int, action="append")
    parser.add_argument("--exploration-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.contexts is not None and args.context_ids:
        raise ValueError("use either --contexts or --context-id, not both")
    refresh_limits = args.seam_refresh_per_prefix or [0, 1, 2]
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
    search = DestinationPlanSearch(od_costs=od_costs, destination_inputs=destination_inputs)

    if args.contexts is None:
        context_ids = args.context_ids or [17956, 75543]
    else:
        selection_args = SimpleNamespace(
            contexts=args.contexts,
            candidate_contexts=args.candidate_contexts,
            max_layers=args.max_layers,
            trace_context=None,
            exploration_seed=args.exploration_seed,
        )
        context_ids = eligible_context_ids(steps, selection_args)
    masses = {refresh_per_prefix: [] for refresh_per_prefix in refresh_limits}
    added_states = {refresh_per_prefix: [] for refresh_per_prefix in refresh_limits}
    skipped = 0

    for context_id in context_ids:
        context_steps = steps.filter(steps["context_id"] == context_id)
        context_initial = initial_locations.filter(
            initial_locations["context_id"] == context_id
        )
        if context_steps.is_empty() or context_initial.is_empty():
            raise ValueError(f"context {context_id} is not present in the prepared inputs")
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
        except ValueError:
            if args.contexts is None:
                raise
            skipped += 1
            continue
        oracle_top_k = ranked_plans(oracle_table)[: args.top_k]
        print(f"\ncontext={context_id}")
        print(
            "  steps="
            f"{[(row['activity_id'], row['anchor_id'], row['fixed_destination']) for row in context_steps.sort('layer').to_dicts()]}"
        )
        for refresh_per_prefix in refresh_limits:
            table, report = search.top_k(
                steps=context_steps,
                initial_locations=context_initial,
                logit_scale=LOGIT_SCALE,
                update_plan_timings=True,
                use_shadow_prices=True,
                exploration_seed=args.exploration_seed,
                frontier_width=args.frontier_width,
                proposal_limit_per_source=args.proposal_limit_per_source,
                stitch_bias=0,
                continuation_state_limit=args.continuation_state_limit,
                continuation_proposal_limit=args.continuation_proposal_limit,
                seam_refresh_per_prefix=refresh_per_prefix,
                top_k=args.top_k,
                n_threads=1,
                skip_infeasible=False,
            )
            plans = ranked_plans(table)
            plan_zones = {zones for zones, _ in plans}
            retained_mass = retained_probability_mass(oracle_top_k, plan_zones)
            masses[refresh_per_prefix].append(retained_mass)
            added_states[refresh_per_prefix].append(report["seam_refresh_states"])
            exact_ranks = [
                rank
                for rank, (zones, _) in enumerate(oracle_top_k, start=1)
                if zones in plan_zones
            ]
            print(
                f"  refresh={refresh_per_prefix}: mass={retained_mass:.4f} "
                f"exact-ranks={exact_ranks} "
                f"states-added={report['seam_refresh_states']} "
                f"proposals={report['seam_refresh_proposals']} "
                f"stitch-pairs={report['stitch_pairs']}"
            )
        if args.contexts is not None and len(masses[refresh_limits[0]]) >= args.contexts:
            break

    if args.contexts is not None:
        proven = len(masses[refresh_limits[0]])
        print(f"\nproven={proven} skipped={skipped}")
        for refresh_per_prefix in refresh_limits:
            print(
                f"  refresh={refresh_per_prefix}: "
                f"mean-mass={sum(masses[refresh_per_prefix]) / proven:.6f} "
                f"mean-states-added="
                f"{sum(added_states[refresh_per_prefix]) / proven:.2f}"
            )


if __name__ == "__main__":
    main()
