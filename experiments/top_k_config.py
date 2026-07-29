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
    "factor_map_max_depth": 5,
    "stitch_bias": 1,
    "continuation_state_limit": 1,
    "deep_continuation_state_limit": 2,
    "continuation_proposal_limit": 1,
    "seam_refresh_per_prefix": 1,
    "pricing_passes": 2,
    "pricing_seed_limit": 10,
    "pricing_column_limit": 4,
    "pricing_pair_candidate_limit": 4,
    "pricing_pair_deep_candidate_limit": 8,
    "pricing_pair_deep_min_layers": 0,
    "pricing_next_pass_min_new": 3,
    "pricing_min_layers": 6,
}

CANDIDATE_STRATEGIES = (
    "factor_map",
    "symmetric_factor_map",
    "adaptive_factor_map",
    "heuristic",
)

_OPTION_HELP = {
    "frontier_width": "partial plans retained at each forward/backward step",
    "proposal_limit_per_source": "destinations proposed from each retained partial plan",
    "symmetric_message_limit": "right-to-left lookahead states for adjacent unknowns",
    "symmetric_state_limit": "partial right-to-left states retained away from the join",
    "symmetric_forward_proposal_limit": "lookahead destinations offered to the forward search",
    "candidate_strategy": "destination-shortlisting policy",
    "factor_map_max_depth": "longest home-bounded tour using factor-map proposals",
    "stitch_bias": "offset from the middle step where forward and backward search join",
    "continuation_state_limit": "right-side states used to rank each forward proposal",
    "deep_continuation_state_limit": "right-side states used on long tours",
    "continuation_proposal_limit": "destinations projected back from each right-side state",
    "seam_refresh_per_prefix": "extra join states proposed from each retained left side",
    "pricing_passes": "complete-plan improvement rounds (historical public name)",
    "pricing_seed_limit": "best complete plans improved in each round",
    "pricing_column_limit": "single-choice replacements retained per plan",
    "pricing_pair_candidate_limit": "replacements per choice in the pair probe",
    "pricing_pair_deep_candidate_limit": "replacements per choice after pair expansion",
    "pricing_pair_deep_min_layers": "0 selects local expansion; >=2 selects by plan depth",
    "pricing_next_pass_min_new": "new surviving plans required for another improvement round",
    "pricing_min_layers": "minimum plan length receiving post-search improvement",
}


def add_top_k_tuning_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the advanced search controls shared by every experiment command."""
    for name, default in ACTIVE_TOP_K_DEFAULTS.items():
        options: dict[str, Any] = {
            "default": default,
            "type": type(default),
            "help": _OPTION_HELP[name],
        }
        if name == "candidate_strategy":
            options["choices"] = CANDIDATE_STRATEGIES
        parser.add_argument(f"--{name.replace('_', '-')}", **options)


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
