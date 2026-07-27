"""Create and validate immutable experiment manifests."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from experiments.harness import (
    ExperimentKind,
    ExperimentManifest,
    ExperimentStage,
)
from experiments.top_k_config import ACTIVE_TOP_K_DEFAULTS


def _parse_change(value: str) -> tuple[str, Any]:
    name, separator, raw = value.partition("=")
    if not separator or name not in ACTIVE_TOP_K_DEFAULTS:
        raise ValueError(f"--change must name an active top-K option: {value}")
    current = ACTIVE_TOP_K_DEFAULTS[name]
    if isinstance(current, int):
        parsed: Any = int(raw)
    elif isinstance(current, float):
        parsed = float(raw)
    else:
        parsed = raw
    return name, parsed


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return repr(value)


def _options_table(label: str, options: dict[str, Any]) -> list[str]:
    return [
        f"[{label}.options]",
        *(f"{name} = {_toml_value(value)}" for name, value in options.items()),
        "",
    ]


def write_draft(args: argparse.Namespace) -> None:
    path = args.path.resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {path}")
    candidate = dict(ACTIVE_TOP_K_DEFAULTS)
    changes = dict(_parse_change(value) for value in args.change)
    candidate.update(changes)
    gates = {
        ExperimentKind.PURE_PERF.value: [
            "min_rust_improvement = 0.03",
            "require_same_output = true",
            "require_same_counters = true",
        ],
        ExperimentKind.QUALITY_RUNTIME.value: [
            "max_wall_regression = 0.15",
            "stratified_mass_delta_min = 0.0",
            "zero_mass_delta_max = 0.0",
        ],
        ExperimentKind.QUALITY_ONLY.value: [
            "stratified_mass_delta_min = 0.0",
            "zero_mass_delta_max = 0.0",
        ],
    }[args.kind]
    throughput_kind = args.kind in {
        ExperimentKind.PURE_PERF.value,
        ExperimentKind.QUALITY_RUNTIME.value,
    }
    selector = "calibrated" if throughput_kind else "stratified"
    lines = [
        "schema_version = 1",
        f"id = {_toml_value(args.identifier)}",
        f"kind = {_toml_value(args.kind)}",
        f"stage = {_toml_value(args.stage)}",
        'hypothesis = "TODO: state the expected measurable outcome"',
        'mechanism = "TODO: state why the change should produce that outcome"',
        'falsifier = "TODO: state what observation would reject the hypothesis"',
        'unknowns = ["TODO: name the most important unresolved assumption"]',
        "",
        "[comparison]",
        f"allowed_differences = {_toml_value(sorted(changes))}",
        "",
        "[cohort]",
        f'name = "{args.identifier}-{args.stage}"',
        f'role = "{args.stage}"',
        "selection_seed = 42",
        f'selector = "{selector}"',
    ]
    if throughput_kind:
        lines.append("contexts = 1000")
    else:
        lines.append("contexts_per_stratum = 10")
    if args.stage == ExperimentStage.VALIDATION.value:
        lines.append('expected_fingerprint = "TODO: lock after a dry run"')
    lines.extend(
        [
            "",
            "[gates]",
            *gates,
            "",
            "[evidence]",
            'quality_artifact = ""',
            "",
            *_options_table("baseline", dict(ACTIVE_TOP_K_DEFAULTS)),
            *_options_table("candidate", candidate),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote draft {path}")
    print("fill the TODO fields, then run experiment validate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("new", help="snapshot active defaults into a draft")
    create.add_argument("path", type=Path)
    create.add_argument("--id", dest="identifier", required=True)
    create.add_argument(
        "--kind",
        choices=[kind.value for kind in ExperimentKind],
        required=True,
    )
    create.add_argument(
        "--stage",
        choices=[stage.value for stage in ExperimentStage],
        default=ExperimentStage.DISCOVERY.value,
    )
    create.add_argument("--change", action="append", default=[], metavar="NAME=VALUE")

    validate = subparsers.add_parser("validate", help="strictly validate one manifest")
    validate.add_argument("path", type=Path)
    show = subparsers.add_parser("show", help="print resolved identity and differences")
    show.add_argument("path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "new":
        write_draft(args)
        return
    manifest = ExperimentManifest.load(args.path)
    print(
        f"{manifest.identifier} | {manifest.kind.value} | {manifest.stage.value} | "
        f"fingerprint={manifest.fingerprint}"
    )
    if args.command == "show":
        print(f"cohort={manifest.cohort['name']}")
        print(f"differences={list(manifest.allowed_differences)}")
        for name in manifest.allowed_differences:
            print(
                f"  {name}: {manifest.baseline[name]!r} -> "
                f"{manifest.candidate[name]!r}"
            )


if __name__ == "__main__":
    main()
