"""Shared active bounded-search defaults for experiment harnesses."""

from __future__ import annotations

import argparse
from typing import Any


ACTIVE_TOP_K_DEFAULTS = {
    "frontier_width": 40,
    "proposal_limit_per_source": 16,
    "symmetric_message_limit": 4,
    "symmetric_state_limit": 4,
    "symmetric_forward_proposal_limit": 20,
    "candidate_strategy": "adaptive_factor_map",
    "surface_bins": 2,
    "factor_map_max_depth": 5,
    "stitch_bias": 1,
    "continuation_state_limit": 1,
    "deep_continuation_state_limit": 2,
    "continuation_log_gap": 0.0,
    "continuation_proposal_limit": 1,
    "seam_refresh_per_prefix": 1,
    "heuristic_reserve_limit": 0,
    "pricing_passes": 2,
    "pricing_seed_limit": 10,
    "pricing_column_limit": 4,
    "pricing_pair_candidate_limit": 4,
    "pricing_pair_deep_candidate_limit": 8,
    "pricing_pair_deep_min_layers": 0,
    "pricing_next_pass_min_new": 3,
    "pricing_min_layers": 6,
}


def add_top_k_tuning_arguments(parser: argparse.ArgumentParser) -> None:
    """Add every bounded-search tuning option with active-baseline defaults."""
    parser.add_argument("--frontier-width", type=int, default=ACTIVE_TOP_K_DEFAULTS["frontier_width"])
    parser.add_argument(
        "--proposal-limit-per-source",
        type=int,
        default=ACTIVE_TOP_K_DEFAULTS["proposal_limit_per_source"],
    )
    parser.add_argument(
        "--symmetric-message-limit",
        type=int,
        default=ACTIVE_TOP_K_DEFAULTS["symmetric_message_limit"],
    )
    parser.add_argument(
        "--symmetric-state-limit",
        type=int,
        default=ACTIVE_TOP_K_DEFAULTS["symmetric_state_limit"],
    )
    parser.add_argument(
        "--symmetric-forward-proposal-limit",
        type=int,
        default=ACTIVE_TOP_K_DEFAULTS["symmetric_forward_proposal_limit"],
    )
    parser.add_argument(
        "--candidate-strategy",
        choices=(
            "surface",
            "factor_map",
            "symmetric_factor_map",
            "adaptive_factor_map",
            "heuristic",
        ),
        default=ACTIVE_TOP_K_DEFAULTS["candidate_strategy"],
        help="bounded proposal policy (default: adaptive_factor_map)",
    )
    parser.add_argument(
        "--surface-bins",
        type=int,
        choices=(2, 4),
        default=ACTIVE_TOP_K_DEFAULTS["surface_bins"],
    )
    parser.add_argument(
        "--factor-map-max-depth",
        type=int,
        default=ACTIVE_TOP_K_DEFAULTS["factor_map_max_depth"],
        help="maximum home-bounded tour depth for factor-map support",
    )
    parser.add_argument("--stitch-bias", type=int, default=ACTIVE_TOP_K_DEFAULTS["stitch_bias"])
    parser.add_argument(
        "--continuation-state-limit",
        type=int,
        default=ACTIVE_TOP_K_DEFAULTS["continuation_state_limit"],
    )
    parser.add_argument(
        "--deep-continuation-state-limit",
        type=int,
        default=ACTIVE_TOP_K_DEFAULTS["deep_continuation_state_limit"],
    )
    parser.add_argument(
        "--continuation-log-gap",
        type=float,
        default=ACTIVE_TOP_K_DEFAULTS["continuation_log_gap"],
    )
    parser.add_argument(
        "--continuation-proposal-limit",
        type=int,
        default=ACTIVE_TOP_K_DEFAULTS["continuation_proposal_limit"],
    )
    parser.add_argument(
        "--seam-refresh-per-prefix",
        type=int,
        default=ACTIVE_TOP_K_DEFAULTS["seam_refresh_per_prefix"],
    )
    parser.add_argument(
        "--heuristic-reserve-limit",
        type=int,
        default=ACTIVE_TOP_K_DEFAULTS["heuristic_reserve_limit"],
        help="add this many heuristic candidates when factor-map and heuristic support share fewer than this many zones",
    )
    parser.add_argument(
        "--pricing-passes",
        type=int,
        default=ACTIVE_TOP_K_DEFAULTS["pricing_passes"],
    )
    parser.add_argument(
        "--pricing-seed-limit",
        type=int,
        default=ACTIVE_TOP_K_DEFAULTS["pricing_seed_limit"],
    )
    parser.add_argument(
        "--pricing-column-limit",
        type=int,
        default=ACTIVE_TOP_K_DEFAULTS["pricing_column_limit"],
    )
    parser.add_argument(
        "--pricing-pair-candidate-limit",
        type=int,
        default=ACTIVE_TOP_K_DEFAULTS["pricing_pair_candidate_limit"],
    )
    parser.add_argument(
        "--pricing-pair-deep-candidate-limit",
        type=int,
        default=ACTIVE_TOP_K_DEFAULTS["pricing_pair_deep_candidate_limit"],
    )
    parser.add_argument(
        "--pricing-pair-deep-min-layers",
        type=int,
        default=ACTIVE_TOP_K_DEFAULTS["pricing_pair_deep_min_layers"],
    )
    parser.add_argument(
        "--pricing-next-pass-min-new",
        type=int,
        default=ACTIVE_TOP_K_DEFAULTS["pricing_next_pass_min_new"],
    )
    parser.add_argument(
        "--pricing-min-layers",
        type=int,
        default=ACTIVE_TOP_K_DEFAULTS["pricing_min_layers"],
    )


def top_k_tuning_options(args: argparse.Namespace) -> dict[str, Any]:
    """Return the shared tuning fields in the names accepted by ``top_k``."""
    return {name: getattr(args, name) for name in ACTIVE_TOP_K_DEFAULTS}


def apply_top_k_overrides(
    baseline: dict[str, Any],
    values: list[str],
) -> dict[str, Any]:
    """Apply compact NAME=VALUE overrides for fast exploratory A/B runs."""
    options = dict(baseline)
    for value in values:
        name, separator, raw = value.partition("=")
        if not separator or name not in options:
            raise ValueError(f"override must name an active top-K option: {value}")
        current = options[name]
        if isinstance(current, int):
            options[name] = int(raw)
        elif isinstance(current, float):
            options[name] = float(raw)
        else:
            options[name] = raw
    return options
