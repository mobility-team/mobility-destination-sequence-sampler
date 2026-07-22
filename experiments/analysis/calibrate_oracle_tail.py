"""Calibrate a top-K score-tail estimate against cached exact top-500 plans.

This estimates the normalizer of a *finite top-500 support*.  It does not
estimate the total probability mass of all feasible paths: that requires either
an exact partition function, a defensible count/bound on the unseen tail, or a
separate sampling estimator.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import mean, median

import polars as pl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oracle-cache",
        type=Path,
        default=Path("experiments/.cache/oracle-top-k/eb755d38afc218332a39"),
    )
    parser.add_argument("--support", type=int, default=500)
    parser.add_argument("--top-k", type=int, action="append", dest="top_ks")
    return parser.parse_args()


def scores_from_table(table: pl.DataFrame) -> list[float]:
    """Read one exact score per draw in descending exact-plan order."""
    return sorted(
        (float(score) for score in table.group_by("draw_id").agg(pl.col("total_log_weight").first())["total_log_weight"]),
        reverse=True,
    )


def normalized_weights(scores: list[float]) -> list[float]:
    maximum = scores[0]
    return [math.exp(score - maximum) for score in scores]


def geometric_tail(weights: list[float], scores: list[float], top_k: int, support: int) -> float:
    """Estimate ranks K+1..support from recent log-score spacings.

    A locally linear score-by-rank curve means weights form a geometric tail.
    This intentionally fails loudly (infinite estimate) if the local spacing is
    non-decreasing rather than silently claiming a normalized total.
    """
    window = min(10, top_k - 1)
    decrements = [
        scores[index + 1] - scores[index]
        for index in range(top_k - window, top_k - 1)
    ]
    log_ratio = mean(decrements)
    ratio = math.exp(log_ratio)
    if ratio >= 1.0:
        return math.inf
    remaining = support - top_k
    if remaining <= 0:
        return 0.0
    return weights[top_k - 1] * ratio * (1.0 - ratio**remaining) / (1.0 - ratio)


def summarize(values: list[float]) -> tuple[float, float, float]:
    ordered = sorted(values)
    return ordered[0], median(ordered), ordered[-1]


def main() -> None:
    args = parse_args()
    top_ks = sorted(set(args.top_ks or [10, 20, 50, 100]))
    if min(top_ks) < 2 or max(top_ks) >= args.support:
        raise ValueError("top-K values must be in [2, support)")
    paths = sorted(args.oracle_cache.glob(f"context-*-k{args.support}-states-*.parquet"))
    if not paths:
        raise ValueError(f"no top-{args.support} oracle tables in {args.oracle_cache}")
    rows: dict[int, list[tuple[float, float, float]]] = {top_k: [] for top_k in top_ks}
    for path in paths:
        scores = scores_from_table(pl.read_parquet(path))
        if len(scores) < args.support:
            continue
        scores = scores[: args.support]
        weights = normalized_weights(scores)
        normalizer = sum(weights)
        for top_k in top_ks:
            observed = sum(weights[:top_k]) / normalizer
            estimated_tail = geometric_tail(weights, scores, top_k, args.support)
            estimated = sum(weights[:top_k]) / (sum(weights[:top_k]) + estimated_tail)
            rows[top_k].append((observed, estimated, estimated - observed))
    print(f"top-{args.support} exact tables: {len(paths)}; complete support tables: {len(rows[top_ks[0]])}")
    print("K | observed top-K mass in top-support | estimate | signed error (min/median/max) | MAE")
    for top_k in top_ks:
        observed = [row[0] for row in rows[top_k]]
        estimated = [row[1] for row in rows[top_k]]
        errors = [row[2] for row in rows[top_k]]
        low, middle, high = summarize(errors)
        mae = mean(abs(error) for error in errors)
        print(
            f"{top_k:3d} | {mean(observed):.3f} | {mean(estimated):.3f} | "
            f"{low:+.3f}/{middle:+.3f}/{high:+.3f} | {mae:.3f}"
        )


if __name__ == "__main__":
    main()
