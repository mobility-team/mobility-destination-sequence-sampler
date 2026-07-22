"""Calibrate head-only exp(U) tail fits on fully enumerable held-out contexts."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean, median

import polars as pl

from mobility_destination_sequence_sampler import DestinationPlanSearch

from experiments.benchmarks.perf_grand_geneve_cache import (
    DEFAULT_GROUP_DAY_TRIPS_FOLDER,
    LOGIT_SCALE,
    prepare_complete_contexts,
    prepare_destination_inputs,
    prepare_od_costs,
    resolve_snapshot_files,
)


CONTEXT_ID = re.compile(r"context-(\d+)-k100-states-")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=int, default=40)
    parser.add_argument("--max-assignments", type=int, default=10_000)
    parser.add_argument("--top-k", type=int, action="append", dest="top_ks")
    parser.add_argument(
        "--oracle-cache-root", type=Path, default=Path("experiments/.cache/oracle-top-k")
    )
    parser.add_argument(
        "--group-day-trips-folder", type=Path, default=DEFAULT_GROUP_DAY_TRIPS_FOLDER
    )
    return parser.parse_args()


def candidate_context_ids(cache_root: Path, max_assignments: int) -> list[int]:
    candidates: dict[int, int] = {}
    for path in cache_root.rglob("*k100-states-*.json"):
        if path.name.endswith(".error.json"):
            continue
        match = CONTEXT_ID.match(path.name)
        if match is None:
            continue
        lattice = int(json.loads(path.read_text())["assignment_lattice"])
        if lattice <= max_assignments:
            context_id = int(match.group(1))
            candidates[context_id] = min(candidates.get(context_id, lattice), lattice)
    return sorted(candidates, key=lambda context_id: ((context_id * 1_103_515_245) % 2**32, context_id))


def regression(values: list[float], features: list[float]) -> tuple[float, float]:
    mean_x, mean_y = mean(features), mean(values)
    denominator = sum((value - mean_x) ** 2 for value in features)
    if denominator == 0.0:
        return mean_y, 0.0
    slope = sum(
        (feature - mean_x) * (value - mean_y)
        for feature, value in zip(features, values, strict=True)
    ) / denominator
    return mean_y - slope * mean_x, slope


def curve_r_squared(scores: list[float], exponent: float | None) -> float:
    """R² for exponential, power-law, or stretched-exponential rank curves."""
    features = [
        math.log(rank) if exponent is None else rank**exponent
        for rank in range(1, len(scores) + 1)
    ]
    intercept, slope = regression(scores, features)
    mean_score = mean(scores)
    residual = sum(
        (score - (intercept + slope * feature)) ** 2
        for score, feature in zip(scores, features, strict=True)
    )
    total = sum((score - mean_score) ** 2 for score in scores)
    return 1.0 - residual / total if total else 1.0


def predicted_mass(scores: list[float], top_k: int, method: str) -> float:
    """Fit head scores, extrapolate through the known finite path count."""
    head_scores = scores[:top_k]
    maximum = scores[0]
    head_weight = sum(math.exp(score - maximum) for score in head_scores)
    if method == "local_geometric":
        window = min(10, top_k - 1)
        slope = mean(
            head_scores[index + 1] - head_scores[index]
            for index in range(top_k - window - 1, top_k - 1)
        )
        intercept = head_scores[-1] - slope * (top_k - 1)
        predict_score = lambda rank: intercept + slope * (rank - 1)
    elif method == "rank_linear":
        intercept, slope = regression(head_scores, list(range(1, top_k + 1)))
        predict_score = lambda rank: intercept + slope * rank
    elif method == "log_rank":
        intercept, slope = regression(head_scores, [math.log(rank) for rank in range(1, top_k + 1)])
        predict_score = lambda rank: intercept + slope * math.log(rank)
    else:
        raise ValueError(method)
    tail_weight = sum(math.exp(predict_score(rank) - maximum) for rank in range(top_k + 1, len(scores) + 1))
    return head_weight / (head_weight + tail_weight)


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] if lower == upper else ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def main() -> None:
    args = parse_args()
    top_ks = sorted(set(args.top_ks or [10, 20, 50, 100]))
    if args.contexts < 4 or args.max_assignments <= max(top_ks):
        raise ValueError("need at least four contexts and max_assignments above every K")
    ids = candidate_context_ids(args.oracle_cache_root, args.max_assignments)[: args.contexts]
    if len(ids) < 4:
        raise RuntimeError("not enough cached small-oracle contexts")
    files = resolve_snapshot_files(args.group_day_trips_folder)
    search = DestinationPlanSearch(
        od_costs=prepare_od_costs(files["transport_costs"], files["demand_groups"]),
        destination_inputs=prepare_destination_inputs(
            files["destination_saturation"], files["demand_groups"]
        ),
    )
    steps, initial_locations, _ = prepare_complete_contexts(
        activity_sequences_path=files["activity_sequences"],
        survey_plan_steps_path=files["survey_plan_steps"],
        demand_groups_path=files["demand_groups"],
        activity_dur_path=files["activity_dur"],
    )
    rows = []
    shape_rows = []
    for context_id in ids:
        distribution = search.exact_distribution(
            steps=steps.filter(pl.col("context_id") == context_id),
            initial_locations=initial_locations.filter(pl.col("context_id") == context_id),
            logit_scale=LOGIT_SCALE,
            update_plan_timings=True,
            use_shadow_prices=True,
            max_assignments=args.max_assignments,
        )
        scores = distribution["scores"]
        maximum = distribution["log_normalizer"]
        weights = [math.exp(score - maximum) for score in scores]
        entropy = -sum(weight * math.log(weight) for weight in weights)
        shape_rows.append(
            {
                "context_id": context_id,
                "feasible": len(scores),
                "top10_mass": sum(weights[:10]),
                "top50_mass": sum(weights[:50]),
                "effective_paths": math.exp(entropy),
                "r2_exponential": curve_r_squared(scores, 1.0),
                "r2_sqrt": curve_r_squared(scores, 0.5),
                "r2_quarter": curve_r_squared(scores, 0.25),
                "r2_power": curve_r_squared(scores, None),
            }
        )
        for top_k in top_ks:
            if top_k >= len(scores):
                continue
            actual = sum(math.exp(score - maximum) for score in scores[:top_k])
            for method in ("local_geometric", "rank_linear", "log_rank"):
                rows.append(
                    {
                        "context_id": context_id,
                        "top_k": top_k,
                        "method": method,
                        "actual": actual,
                        "predicted": predicted_mass(scores, top_k, method),
                    }
                )
    frame = pl.DataFrame(rows).with_columns(error=pl.col("actual") - pl.col("predicted"))
    train = frame.filter(pl.col("context_id") % 2 == 0)
    test = frame.filter(pl.col("context_id") % 2 == 1)
    print(f"fully enumerated contexts={len(ids)}; rows={frame.height}; train/test={train['context_id'].n_unique()}/{test['context_id'].n_unique()}")
    shapes = pl.DataFrame(shape_rows)
    print(
        "shape medians: "
        f"feasible={shapes['feasible'].median():.0f}, "
        f"effective={shapes['effective_paths'].median():.1f}, "
        f"top10-mass={shapes['top10_mass'].median():.3f}, "
        f"top50-mass={shapes['top50_mass'].median():.3f}, "
        f"R²(exp/sqrt/quarter/power)="
        f"{shapes['r2_exponential'].median():.3f}/"
        f"{shapes['r2_sqrt'].median():.3f}/"
        f"{shapes['r2_quarter'].median():.3f}/"
        f"{shapes['r2_power'].median():.3f}"
    )
    winners = shapes.select(
        pl.concat_list("r2_exponential", "r2_sqrt", "r2_quarter", "r2_power")
        .list.arg_max()
        .alias("winner")
    )["winner"].to_list()
    names = ("exponential", "stretched sqrt", "stretched quarter", "power law")
    print(
        "best full-curve R² family counts: "
        + ", ".join(f"{names[index]}={winners.count(index)}" for index in range(len(names)))
    )
    print("K | model | test MAE | median error | empirical 80% interval | test 80% coverage")
    for top_k in top_ks:
        for method in ("local_geometric", "rank_linear", "log_rank"):
            train_errors = train.filter((pl.col("top_k") == top_k) & (pl.col("method") == method))["error"].to_list()
            test_rows = test.filter((pl.col("top_k") == top_k) & (pl.col("method") == method)
            )
            if not train_errors or test_rows.is_empty():
                continue
            low, high = quantile(train_errors, 0.1), quantile(train_errors, 0.9)
            errors = test_rows["error"].to_list()
            coverage = sum(low <= error <= high for error in errors) / len(errors)
            print(
                f"{top_k:3d} | {method:15s} | {mean(abs(error) for error in errors):.3f} | "
                f"{median(errors):+.3f} | [{low:+.3f}, {high:+.3f}] | {coverage:.0%}"
            )


if __name__ == "__main__":
    main()
