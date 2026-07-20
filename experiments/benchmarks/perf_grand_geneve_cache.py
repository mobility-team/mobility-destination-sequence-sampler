"""Grand Geneve cache paths and columnar input preparation for active benchmarks."""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl


DEFAULT_GROUP_DAY_TRIPS_FOLDER = Path(
    r"D:\data\mobility\projects\grand-geneve\group_day_trips"
)

# Coherent iteration-5 snapshot written by the same Grand Genève run on
# 2026-07-15. The benchmark only reads these cache files.
SNAPSHOT_FILES = {
    "activity_sequences": (
        "activity-sequences/"
        "31568cd267580e15713a8458b22b0687-activity_sequences_5.parquet"
    ),
    "transport_costs": (
        "iteration-transport-costs/"
        "0d1f446fa31903f5e734195585678c76-transport_costs_5.parquet"
    ),
    "destination_saturation": (
        "iteration-state-cache/"
        "50489de7b4b351be3778dad7894caef0-destination_saturation_5.parquet"
    ),
    "activity_dur": (
        "iteration-state-cache/"
        "50489de7b4b351be3778dad7894caef0-activity_dur_5.parquet"
    ),
    "demand_groups": (
        "iteration-state-cache/"
        "50489de7b4b351be3778dad7894caef0-demand_groups_5.parquet"
    ),
    "survey_plan_steps": (
        "iteration-state-cache/"
        "50489de7b4b351be3778dad7894caef0-survey_plan_steps_5.parquet"
    ),
}

ACTIVITY_IDS = {
    "home": 0,
    "work": 1,
    "studies": 2,
    "shopping": 3,
    "leisure": 4,
    "other": 5,
}

# Values used by D:\dev\mobility-grand-geneve\run_scenarios.py.
VALUE_OF_TIME = {
    "home": 3.5,
    "work": 2.0,
    "studies": 3.0,
    "shopping": 8.0,
    "leisure": 4.0,
    "other": 5.0,
}

LOGIT_SCALE = 0.25
MIN_ACTIVITY_TIME_CONSTANT = 2.0


def resolve_snapshot_files(folder: Path) -> dict[str, Path]:
    files = {
        name: folder / relative_path
        for name, relative_path in SNAPSHOT_FILES.items()
    }
    missing = [path for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Grand Genève cache files are missing:\n"
            + "\n".join(str(path) for path in missing)
        )
    return files


def prepare_od_costs(
    path: Path,
    demand_groups_path: Path,
) -> pl.DataFrame:
    # Match IterationChoiceInputs.destination_od_costs: modes close to the
    # cheapest mode get more weight, and cost/time are averaged together.
    mode_costs = pl.scan_parquet(path).select(
        pl.col("from").cast(pl.UInt32).alias("origin"),
        pl.col("to").cast(pl.UInt32).alias("destination"),
        "cost",
        "time",
    )
    minimum_costs = mode_costs.group_by(["origin", "destination"]).agg(
        minimum_cost=pl.col("cost").min()
    )
    od_costs = (
        mode_costs.join(minimum_costs, on=["origin", "destination"])
        .with_columns(
            mode_weight=(
                -pl.lit(LOGIT_SCALE)
                * (pl.col("cost") - pl.col("minimum_cost"))
            ).exp()
        )
        .group_by(["origin", "destination"])
        .agg(
            weighted_cost=(pl.col("mode_weight") * pl.col("cost")).sum(),
            weighted_time=(pl.col("mode_weight") * pl.col("time")).sum(),
            total_weight=pl.col("mode_weight").sum(),
        )
        .select(
            "origin",
            "destination",
            cost=pl.col("weighted_cost") / pl.col("total_weight"),
            time=pl.col("weighted_time") / pl.col("total_weight"),
        )
        .collect(engine="streaming")
    )

    # Every zone must retain Mobility's modeled within-zone trip. A missing or
    # zero diagonal would hide an input-generation problem and would make
    # same-zone destinations artificially attractive in the benchmarks.
    zones = pl.concat(
        [
            od_costs.select(pl.col("origin").alias("zone")),
            od_costs.select(pl.col("destination").alias("zone")),
            pl.read_parquet(demand_groups_path).select(
                pl.col("home_zone_id").cast(pl.UInt32).alias("zone")
            ),
        ]
    ).unique()
    diagonal = od_costs.filter(
        pl.col("origin") == pl.col("destination")
    )
    missing_diagonal = zones.join(
        diagonal.select(pl.col("origin").alias("zone")),
        on="zone",
        how="anti",
    )
    if missing_diagonal.height > 0:
        raise ValueError(
            "Transport costs are missing within-zone entries for "
            f"{missing_diagonal.height} zones; first IDs: "
            f"{missing_diagonal['zone'].head(10).to_list()}"
        )
    invalid_diagonal = diagonal.filter(
        ~pl.col("cost").is_finite()
        | ~pl.col("time").is_finite()
        | (pl.col("cost") <= 0.0)
        | (pl.col("time") <= 0.0)
    )
    if invalid_diagonal.height > 0:
        raise ValueError(
            "Transport costs contain nonpositive or nonfinite within-zone "
            f"entries for {invalid_diagonal.height} zones; first rows: "
            f"{invalid_diagonal.head(10).to_dicts()}"
        )
    return od_costs.sort(["origin", "destination"])


def prepare_destination_inputs(
    saturation_path: Path,
    demand_groups_path: Path,
) -> pl.DataFrame:
    # Home zones cover the Grand Genève zone set and provide the destination
    # country needed by the work activity's CH coefficient.
    zone_countries = (
        pl.scan_parquet(demand_groups_path)
        .select(
            pl.col("home_zone_id").cast(pl.UInt32).alias("destination"),
            pl.col("country").cast(pl.Utf8).alias("destination_country"),
        )
        .unique()
    )
    return (
        pl.scan_parquet(saturation_path)
        .select(
            pl.col("activity").cast(pl.Utf8),
            pl.col("to").cast(pl.UInt32).alias("destination"),
            "opportunity_capacity",
            pl.col("k_saturation_utility").alias("saturation_utility"),
            pl.col("destination_shadow_price").alias("shadow_price"),
        )
        .join(zone_countries, on="destination", how="left")
        .with_columns(
            activity_id=pl.col("activity").replace_strict(
                ACTIVITY_IDS,
                return_dtype=pl.UInt32,
            ),
            country_value_coefficient=(
                pl.when(
                    (pl.col("activity") == "work")
                    & (pl.col("destination_country") == "ch")
                )
                .then(1.5)
                .otherwise(1.0)
            ),
        )
        .select(
            "activity_id",
            "destination",
            "opportunity_capacity",
            "country_value_coefficient",
            "saturation_utility",
            "shadow_price",
        )
        .collect(engine="streaming")
    )


def prepare_complete_contexts(
    *,
    activity_sequences_path: Path,
    survey_plan_steps_path: Path,
    demand_groups_path: Path,
    activity_dur_path: Path,
) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    demand_unit_cols = ["demand_group_id", "demand_subgroup_id"]
    sequence_cols = ["activity_seq_id", "time_seq_id"]
    raw_context_cols = demand_unit_cols + sequence_cols

    demand_groups = pl.scan_parquet(demand_groups_path).select(
        demand_unit_cols
        + [
            pl.col("home_zone_id").cast(pl.UInt32),
            pl.col("country").cast(pl.Utf8),
            pl.col("csp").cast(pl.Utf8),
        ]
    )
    activity_durations = pl.scan_parquet(activity_dur_path).select(
        pl.col("country").cast(pl.Utf8),
        pl.col("csp").cast(pl.Utf8),
        pl.col("activity").cast(pl.Utf8),
        "mean_duration_per_pers",
    )
    survey_durations = pl.scan_parquet(survey_plan_steps_path).select(
        sequence_cols + ["seq_step_index", "duration_per_pers"]
    )

    steps = (
        pl.scan_parquet(activity_sequences_path)
        .filter(pl.col("activity_seq_id") != 0)
        .with_columns(activity=pl.col("activity").cast(pl.Utf8))
        .join(demand_groups, on=demand_unit_cols)
        .join(
            survey_durations,
            on=sequence_cols + ["seq_step_index"],
            how="left",
        )
        .join(
            activity_durations,
            on=["country", "csp", "activity"],
            how="inner",
        )
        .with_columns(
            activity_id=pl.col("activity").replace_strict(
                ACTIVITY_IDS,
                return_dtype=pl.UInt32,
            ),
            anchor_id=(
                pl.when(
                    pl.col("is_anchor")
                    & (pl.col("activity") != "home")
                )
                .then(
                    pl.col("activity").replace_strict(
                        ACTIVITY_IDS,
                        return_dtype=pl.UInt32,
                    )
                )
                .otherwise(pl.lit(None, dtype=pl.UInt32))
            ),
            value_of_time=pl.col("activity").replace_strict(
                VALUE_OF_TIME,
                return_dtype=pl.Float64,
            ),
            min_activity_time=(
                pl.col("mean_duration_per_pers")
                * math.exp(-MIN_ACTIVITY_TIME_CONSTANT)
            ),
            fixed_destination=(
                pl.when(pl.col("activity") == "home")
                .then(pl.col("home_zone_id"))
                .otherwise(pl.lit(None, dtype=pl.UInt32))
            ),
            duration_per_person=pl.col("duration_per_pers").fill_null(
                pl.col("next_departure_time") - pl.col("arrival_time")
            ),
        )
        .sort(raw_context_cols + ["seq_step_index"])
        .with_columns(
            layer=pl.int_range(pl.len())
            .over(raw_context_cols)
            .cast(pl.UInt32)
        )
        .with_columns(
            # The terminal home row is a day boundary, not a fixed-arrival
            # activity. Return-home travel can therefore consume overnight
            # home time instead of pushing every delay into the prior activity.
            arrival_time_rigidity=(
                pl.when(
                    pl.col("layer")
                    == pl.col("layer").max().over(raw_context_cols)
                )
                .then(pl.lit(0.0))
                .otherwise(pl.col("is_anchor").cast(pl.Float64))
            )
        )
        .with_columns(
            origin_activity=(
                pl.col("activity")
                .shift(1)
                .over(raw_context_cols)
                .fill_null(pl.lit("home"))
            ),
            origin_is_anchor=(
                pl.col("is_anchor")
                .shift(1)
                .over(raw_context_cols)
                .fill_null(False)
            ),
        )
        .with_columns(
            # Every trip leaves the previous activity. Home stays flexible;
            # other anchor activities keep the default rigid departure.
            departure_time_rigidity=(
                pl.when(pl.col("origin_activity") == "home")
                .then(pl.lit(0.0))
                .otherwise(pl.col("origin_is_anchor").cast(pl.Float64))
            )
        )
        .select(
            raw_context_cols
            + [
                "home_zone_id",
                "country",
                "csp",
                "layer",
                "activity_id",
                "anchor_id",
                "fixed_destination",
                pl.col("departure_time").cast(pl.Float64),
                pl.col("arrival_time").cast(pl.Float64),
                "arrival_time_rigidity",
                "departure_time_rigidity",
                pl.col("next_departure_time").cast(pl.Float64),
                pl.col("duration_per_person").cast(pl.Float64),
                "value_of_time",
                pl.col("mean_duration_per_pers").alias(
                    "mean_duration_per_person"
                ),
                "min_activity_time",
            ]
        )
        .collect(engine="streaming")
    )

    raw_context_count = steps.select(raw_context_cols).unique().height

    # Polars deduplicates complete recursion inputs. Hashing each ordered step
    # keeps this preparation compact without expanding any destination choices.
    step_value_cols = [
        "layer",
        "activity_id",
        "anchor_id",
        "fixed_destination",
        "departure_time",
        "arrival_time",
        "arrival_time_rigidity",
        "departure_time_rigidity",
        "next_departure_time",
        "duration_per_person",
        "value_of_time",
        "mean_duration_per_person",
        "min_activity_time",
    ]
    profiles = (
        steps.with_columns(
            step_hash=pl.struct(step_value_cols).hash(seed=17)
        )
        .group_by(raw_context_cols + ["home_zone_id", "country", "csp"])
        .agg(
            sequence_hash=pl.col("step_hash")
            .sort_by("layer")
            .cast(pl.String)
            .str.join("-")
        )
        .with_columns(
            profile_key=pl.concat_str(
                [
                    pl.col("home_zone_id").cast(pl.String),
                    "country",
                    "csp",
                    "sequence_hash",
                ],
                separator="|",
            )
        )
    )
    unique_profiles = (
        profiles.select("profile_key")
        .unique()
        .sort("profile_key")
        .with_row_index("context_id")
        .with_columns(pl.col("context_id").cast(pl.UInt64))
    )
    raw_to_context = profiles.join(unique_profiles, on="profile_key").select(
        raw_context_cols + ["context_id"]
    )
    unique_steps = (
        steps.join(raw_to_context, on=raw_context_cols)
        .sort(["context_id", "layer"])
        .unique(["context_id", "layer"], keep="first")
        .select(
            "context_id",
            "layer",
            "activity_id",
            "anchor_id",
            "fixed_destination",
            "departure_time",
            "arrival_time",
            "arrival_time_rigidity",
            "departure_time_rigidity",
            "next_departure_time",
            "duration_per_person",
            "value_of_time",
            "mean_duration_per_person",
            "min_activity_time",
        )
    )
    initial_locations = (
        steps.join(raw_to_context, on=raw_context_cols)
        .select(
            "context_id",
            pl.col("home_zone_id").alias("initial_zone"),
        )
        .unique()
        .sort("context_id")
    )
    return unique_steps, initial_locations, raw_context_count


def select_contexts(
    steps: pl.DataFrame,
    initial_locations: pl.DataFrame,
    n_contexts: int,
    *,
    seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    selected_ids = (
        steps.group_by("context_id")
        .agg(n_layers=pl.len())
        .with_columns(
            sample_order=pl.col("context_id").hash(seed=seed)
        )
        .sort("sample_order")
        .head(n_contexts)
        .select("context_id")
    )
    return (
        steps.join(selected_ids, on="context_id", how="semi"),
        initial_locations.join(selected_ids, on="context_id", how="semi"),
    )
