from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from mobility_destination_sequence_sampler._core import (
    ExperimentalDestinationSampler as DestinationSampler,
)
from scipy.cluster.vq import kmeans2
from scipy.sparse.linalg import svds

from experiments.benchmarks.perf_grand_geneve_cache import (
    DEFAULT_GROUP_DAY_TRIPS_FOLDER,
    LOGIT_SCALE,
    prepare_complete_contexts,
    prepare_destination_inputs,
    prepare_od_costs,
    resolve_snapshot_files,
)


MIN_ACTIVITY_DURATION_HOURS = 1e-3


@dataclass(frozen=True)
class FineInputs:
    zone_ids: np.ndarray
    costs: np.ndarray
    times: np.ndarray
    activity_ids: np.ndarray
    capacity: np.ndarray
    country_coefficient: np.ndarray
    shadow_price: np.ndarray


@dataclass(frozen=True)
class CoarseInputs:
    labels: np.ndarray
    costs: np.ndarray
    times: np.ndarray
    capacity: np.ndarray
    country_coefficient: np.ndarray
    shadow_price: np.ndarray


@dataclass(frozen=True)
class ContextResult:
    log_partition: float
    first_destination_probability: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the exact rigidity-aware backward-forward algorithm after "
            "aggregating the Grand Geneve transport zones."
        )
    )
    parser.add_argument(
        "--group-day-trips-folder",
        type=Path,
        default=DEFAULT_GROUP_DAY_TRIPS_FOLDER,
    )
    parser.add_argument(
        "--clusters",
        type=int,
        nargs="+",
        default=[16, 32, 64],
    )
    parser.add_argument("--embedding-rank", type=int, default=32)
    parser.add_argument("--n-contexts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument(
        "--wrapped-home-shadow-price",
        type=float,
        default=2.0,
        help="Utility cost of one hour removed from wrapped overnight home time.",
    )
    parser.add_argument(
        "--bidirectional-feasibility",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Intersect a forward exact-feasibility pass with the backward "
            "utility recursion."
        ),
    )
    parser.add_argument("--n-threads", type=int, default=None)
    return parser.parse_args()


def dense_inputs(
    od_costs: pl.DataFrame,
    destination_inputs: pl.DataFrame,
) -> FineInputs:
    zone_ids = np.unique(
        np.concatenate(
            [
                od_costs["origin"].to_numpy(),
                od_costs["destination"].to_numpy(),
            ]
        )
    )
    zone_index = {
        int(zone_id): index for index, zone_id in enumerate(zone_ids)
    }
    origins = np.fromiter(
        (zone_index[int(value)] for value in od_costs["origin"]),
        dtype=np.intp,
        count=od_costs.height,
    )
    destinations = np.fromiter(
        (zone_index[int(value)] for value in od_costs["destination"]),
        dtype=np.intp,
        count=od_costs.height,
    )
    costs = np.full((zone_ids.size, zone_ids.size), np.inf)
    times = np.full((zone_ids.size, zone_ids.size), np.inf)
    costs[origins, destinations] = od_costs["cost"].to_numpy()
    times[origins, destinations] = od_costs["time"].to_numpy()

    activity_ids = np.unique(
        np.concatenate(
            [
                np.asarray([0], dtype=np.uint32),
                destination_inputs["activity_id"].unique().to_numpy(),
            ]
        )
    )
    activity_index = {
        int(activity_id): index
        for index, activity_id in enumerate(activity_ids)
    }
    shape = (activity_ids.size, zone_ids.size)
    capacity = np.zeros(shape)
    country_coefficient = np.ones(shape)
    shadow_price = np.zeros(shape)
    for row in destination_inputs.iter_rows(named=True):
        destination = zone_index.get(int(row["destination"]))
        if destination is None:
            continue
        activity = activity_index[int(row["activity_id"])]
        capacity[activity, destination] = max(
            float(row["opportunity_capacity"]),
            0.0,
        )
        country_coefficient[activity, destination] = float(
            row["country_value_coefficient"]
        )
        shadow_price[activity, destination] = float(row["shadow_price"])

    return FineInputs(
        zone_ids=zone_ids.astype(np.uint32, copy=False),
        costs=costs,
        times=times,
        activity_ids=activity_ids,
        capacity=capacity,
        country_coefficient=country_coefficient,
        shadow_price=shadow_price,
    )


def travel_kernel_embedding(
    costs: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    kernel = np.zeros_like(costs)
    finite = np.isfinite(costs)
    kernel[finite] = np.exp(-LOGIT_SCALE * costs[finite])
    rank = min(rank, kernel.shape[0] - 1)
    left, singular_values, right_transpose = svds(
        kernel,
        k=rank,
        which="LM",
        return_singular_vectors=True,
        random_state=17,
    )
    order = np.argsort(singular_values)[::-1]
    singular_values = singular_values[order]
    left = left[:, order]
    right = right_transpose[order].T
    scale = np.sqrt(singular_values)
    embedding = np.column_stack([left * scale, right * scale])
    standard_deviation = embedding.std(axis=0)
    useful = standard_deviation > 1e-12
    embedding = (
        embedding[:, useful] - embedding[:, useful].mean(axis=0)
    ) / standard_deviation[useful]
    return kernel, embedding


def cluster_zones(
    embedding: np.ndarray,
    n_clusters: int,
    seed: int,
) -> np.ndarray:
    _, labels = kmeans2(
        embedding,
        n_clusters,
        iter=50,
        minit="++",
        missing="raise",
        seed=seed + n_clusters,
    )
    unique = np.unique(labels)
    if unique.size != n_clusters:
        raise ValueError(
            f"k-means returned {unique.size} nonempty clusters, "
            f"expected {n_clusters}"
        )
    remap = {int(label): index for index, label in enumerate(unique)}
    return np.fromiter(
        (remap[int(label)] for label in labels),
        dtype=np.intp,
        count=labels.size,
    )


def aggregate_inputs(
    fine: FineInputs,
    kernel: np.ndarray,
    labels: np.ndarray,
) -> CoarseInputs:
    n_clusters = int(labels.max()) + 1
    membership = np.zeros((labels.size, n_clusters))
    membership[np.arange(labels.size), labels] = 1.0
    cluster_sizes = membership.sum(axis=0)

    kernel_sum = membership.T @ kernel @ membership
    pair_count = np.outer(cluster_sizes, cluster_sizes)
    coarse_kernel = kernel_sum / pair_count
    costs = np.full_like(coarse_kernel, np.inf)
    available = coarse_kernel > 0.0
    costs[available] = -np.log(coarse_kernel[available]) / LOGIT_SCALE

    weighted_times = np.zeros_like(fine.times)
    finite_times = np.isfinite(fine.times)
    weighted_times[finite_times] = (
        kernel[finite_times] * fine.times[finite_times]
    )
    time_numerator = membership.T @ weighted_times @ membership
    times = np.full_like(coarse_kernel, np.inf)
    times[available] = time_numerator[available] / kernel_sum[available]

    capacity = fine.capacity @ membership

    def capacity_weighted(values: np.ndarray, default: float) -> np.ndarray:
        numerator = (fine.capacity * values) @ membership
        result = np.full_like(capacity, default)
        positive = capacity > 0.0
        result[positive] = numerator[positive] / capacity[positive]
        return result

    return CoarseInputs(
        labels=labels,
        costs=costs,
        times=times,
        capacity=capacity,
        country_coefficient=capacity_weighted(
            fine.country_coefficient,
            1.0,
        ),
        shadow_price=capacity_weighted(fine.shadow_price, 0.0),
    )


def logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    finite = np.isfinite(maximum)
    shifted = np.zeros_like(values)
    np.subtract(values, maximum, out=shifted, where=finite)
    exponentials = np.zeros_like(values)
    np.exp(shifted, out=exponentials, where=np.isfinite(values) & finite)
    total = exponentials.sum(axis=axis, keepdims=True)
    result = np.full_like(maximum, -np.inf)
    np.log(total, out=result, where=total > 0.0)
    result += np.where(finite, maximum, 0.0)
    if axis is None:
        return result.reshape(())
    return np.squeeze(result, axis=axis)


def layer_attraction(
    *,
    step: dict[str, float | int | None],
    destinations: np.ndarray,
    activity: int,
    first_choice: bool,
    coarse: CoarseInputs,
) -> np.ndarray:
    if not first_choice:
        return np.zeros(destinations.size)
    capacity = coarse.capacity[activity, destinations]
    attraction = np.full(destinations.size, -np.inf)
    positive = capacity > 0.0
    attraction[positive] = np.log(capacity[positive])
    return attraction


def adjusted_arrival(
    step: dict[str, float | int | None],
    edge_time: np.ndarray,
) -> np.ndarray:
    reference = np.clip(
        float(step["arrival_time"]) - float(step["departure_time"]),
        0.0,
        24.0,
    )
    delta = np.where(
        np.isfinite(edge_time),
        edge_time - reference,
        0.0,
    )
    return (
        float(step["arrival_time"])
        + (1.0 - float(step["arrival_time_rigidity"])) * delta
    )


def adjusted_departure(
    step: dict[str, float | int | None],
    edge_time: np.ndarray,
) -> np.ndarray:
    reference = np.clip(
        float(step["arrival_time"]) - float(step["departure_time"]),
        0.0,
        24.0,
    )
    delta = np.where(
        np.isfinite(edge_time),
        edge_time - reference,
        0.0,
    )
    return (
        float(step["departure_time"])
        - float(step["arrival_time_rigidity"]) * delta
    )


def local_last(
    *,
    step: dict[str, float | int | None],
    previous: np.ndarray,
    current: np.ndarray,
    activity: int,
    first_choice: bool,
    coarse: CoarseInputs,
) -> np.ndarray:
    edge_cost = coarse.costs[np.ix_(previous, current)]
    edge_time = coarse.times[np.ix_(previous, current)]
    arrival = adjusted_arrival(step, edge_time)
    raw_duration = (
        float(step["next_departure_time"]) - arrival
    )
    duration = np.maximum(raw_duration, MIN_ACTIVITY_DURATION_HOURS)
    coefficient = (
        coarse.country_coefficient[activity, current]
        * float(step["value_of_time"])
        + coarse.shadow_price[activity, current]
    )
    duration_factor = np.maximum(
        np.log(duration / float(step["min_activity_time"])),
        0.0,
    )
    utility = (
        coefficient[np.newaxis, :]
        * float(step["mean_duration_per_person"])
        * duration_factor
    )
    local = (
        layer_attraction(
            step=step,
            destinations=current,
            activity=activity,
            first_choice=first_choice,
            coarse=coarse,
        )[np.newaxis, :]
        + LOGIT_SCALE * (utility - edge_cost)
    )
    duration_feasible = (
        raw_duration >= 0.0
        if int(step["activity_id"]) == 0
        else raw_duration > 0.0
    )
    local[
        ~np.isfinite(edge_cost)
        | ~np.isfinite(edge_time)
        | ~duration_feasible
    ] = -np.inf
    return local


def local_with_next(
    *,
    step: dict[str, float | int | None],
    next_step: dict[str, float | int | None],
    previous: np.ndarray,
    current: np.ndarray,
    following: np.ndarray,
    activity: int,
    first_choice: bool,
    coarse: CoarseInputs,
) -> np.ndarray:
    incoming_cost = coarse.costs[np.ix_(previous, current)]
    incoming_time = coarse.times[np.ix_(previous, current)]
    outgoing_time = coarse.times[np.ix_(current, following)]
    arrival = adjusted_arrival(step, incoming_time)
    departure = adjusted_departure(next_step, outgoing_time)
    raw_duration = (
        departure[np.newaxis, :, :] - arrival[:, :, np.newaxis]
    )
    duration = np.maximum(raw_duration, MIN_ACTIVITY_DURATION_HOURS)
    coefficient = (
        coarse.country_coefficient[activity, current]
        * float(step["value_of_time"])
        + coarse.shadow_price[activity, current]
    )
    duration_factor = np.maximum(
        np.log(duration / float(step["min_activity_time"])),
        0.0,
    )
    utility = (
        coefficient[np.newaxis, :, np.newaxis]
        * float(step["mean_duration_per_person"])
        * duration_factor
    )
    local = (
        layer_attraction(
            step=step,
            destinations=current,
            activity=activity,
            first_choice=first_choice,
            coarse=coarse,
        )[np.newaxis, :, np.newaxis]
        + LOGIT_SCALE
        * (utility - incoming_cost[:, :, np.newaxis])
    )
    duration_feasible = (
        raw_duration >= 0.0
        if int(step["activity_id"]) == 0
        else raw_duration > 0.0
    )
    valid = (
        np.isfinite(incoming_cost)[:, :, np.newaxis]
        & np.isfinite(incoming_time)[:, :, np.newaxis]
        & np.isfinite(outgoing_time)[np.newaxis, :, :]
        & duration_feasible
    )
    local[~valid] = -np.inf
    return local


def backward_values(
    *,
    steps: list[dict[str, float | int | None]],
    initial_cluster: int,
    domains: list[np.ndarray],
    activities: list[int],
    first_choices: list[bool],
    coarse: CoarseInputs,
) -> list[np.ndarray]:
    values: list[np.ndarray] = [
        np.empty((0, 0)) for _ in steps
    ]
    previous = (
        domains[-2] if len(domains) > 1
        else np.asarray([initial_cluster], dtype=np.intp)
    )
    values[-1] = local_last(
        step=steps[-1],
        previous=previous,
        current=domains[-1],
        activity=activities[-1],
        first_choice=first_choices[-1],
        coarse=coarse,
    )
    for layer in range(len(steps) - 2, -1, -1):
        previous = (
            np.asarray([initial_cluster], dtype=np.intp)
            if layer == 0
            else domains[layer - 1]
        )
        local = local_with_next(
            step=steps[layer],
            next_step=steps[layer + 1],
            previous=previous,
            current=domains[layer],
            following=domains[layer + 1],
            activity=activities[layer],
            first_choice=first_choices[layer],
            coarse=coarse,
        )
        values[layer] = logsumexp(
            local + values[layer + 1][np.newaxis, :, :],
            axis=2,
        )
    return values


def conditional_result(
    *,
    steps: list[dict[str, float | int | None]],
    initial_cluster: int,
    domains: list[np.ndarray],
    activities: list[int],
    first_choices: list[bool],
    coarse: CoarseInputs,
) -> ContextResult:
    values = backward_values(
        steps=steps,
        initial_cluster=initial_cluster,
        domains=domains,
        activities=activities,
        first_choices=first_choices,
        coarse=coarse,
    )
    first_values = values[0][0]
    log_partition = float(logsumexp(first_values))
    probability = np.zeros(coarse.costs.shape[0])
    if math.isfinite(log_partition):
        probability[domains[0]] = np.exp(
            first_values - log_partition
        )
    return ContextResult(log_partition, probability)


def solve_context(
    *,
    steps: list[dict[str, float | int | None]],
    initial_cluster: int,
    zone_to_cluster: dict[int, int],
    activity_index: dict[int, int],
    coarse: CoarseInputs,
) -> ContextResult | None:
    anchor_ids = sorted(
        {
            int(step["anchor_id"])
            for step in steps
            if step["anchor_id"] is not None
        }
    )
    if len(anchor_ids) > 1:
        return None
    activities = [
        activity_index[int(step["activity_id"])] for step in steps
    ]
    first_choices = []
    seen_anchors: set[int] = set()
    for step in steps:
        anchor_id = step["anchor_id"]
        if step["fixed_destination"] is not None:
            first_choices.append(False)
        elif anchor_id is None:
            first_choices.append(True)
        else:
            anchor = int(anchor_id)
            first_choices.append(anchor not in seen_anchors)
            seen_anchors.add(anchor)

    def domains_for(anchor_cluster: int | None) -> list[np.ndarray]:
        domains = []
        for step, activity in zip(steps, activities, strict=True):
            fixed = step["fixed_destination"]
            if fixed is not None:
                domains.append(
                    np.asarray(
                        [zone_to_cluster[int(fixed)]],
                        dtype=np.intp,
                    )
                )
            elif step["anchor_id"] is not None:
                assert anchor_cluster is not None
                domains.append(
                    np.asarray([anchor_cluster], dtype=np.intp)
                )
            else:
                domains.append(
                    np.flatnonzero(coarse.capacity[activity] > 0.0)
                )
        return domains

    if not anchor_ids:
        return conditional_result(
            steps=steps,
            initial_cluster=initial_cluster,
            domains=domains_for(None),
            activities=activities,
            first_choices=first_choices,
            coarse=coarse,
        )

    anchor_activity = next(
        activity
        for step, activity in zip(steps, activities, strict=True)
        if step["anchor_id"] is not None
    )
    anchor_domain = np.flatnonzero(
        coarse.capacity[anchor_activity] > 0.0
    )
    conditional = [
        conditional_result(
            steps=steps,
            initial_cluster=initial_cluster,
            domains=domains_for(int(anchor_cluster)),
            activities=activities,
            first_choices=first_choices,
            coarse=coarse,
        )
        for anchor_cluster in anchor_domain
    ]
    log_partitions = np.asarray(
        [result.log_partition for result in conditional]
    )
    total = float(logsumexp(log_partitions))
    if not math.isfinite(total):
        return ContextResult(total, np.zeros(coarse.costs.shape[0]))
    weights = np.exp(log_partitions - total)
    first_probability = sum(
        weight * result.first_destination_probability
        for weight, result in zip(weights, conditional, strict=True)
    )
    return ContextResult(total, first_probability)


def rust_tables(
    *,
    context_ids: list[int],
    steps: pl.DataFrame,
    initial_locations: pl.DataFrame,
    fine: FineInputs,
    coarse: CoarseInputs,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    n_clusters = coarse.costs.shape[0]
    origins, destinations = np.nonzero(
        np.isfinite(coarse.costs) & np.isfinite(coarse.times)
    )
    od_costs = pl.DataFrame(
        {
            "origin": origins.astype(np.uint32),
            "destination": destinations.astype(np.uint32),
            "cost": coarse.costs[origins, destinations],
            "time": coarse.times[origins, destinations],
        }
    )

    activity_indices, cluster_indices = np.nonzero(coarse.capacity > 0.0)
    destination_inputs = pl.DataFrame(
        {
            "activity_id": fine.activity_ids[activity_indices].astype(
                np.uint32
            ),
            "destination": cluster_indices.astype(np.uint32),
            "opportunity_capacity": coarse.capacity[
                activity_indices, cluster_indices
            ],
            "country_value_coefficient": coarse.country_coefficient[
                activity_indices, cluster_indices
            ],
            "saturation_utility": np.ones(activity_indices.size),
            "shadow_price": coarse.shadow_price[
                activity_indices, cluster_indices
            ],
        }
    )

    old_zones = fine.zone_ids.tolist()
    new_clusters = coarse.labels.astype(np.uint32).tolist()
    selected_steps = (
        steps.filter(pl.col("context_id").is_in(context_ids))
        .with_columns(
            fixed_destination=pl.col("fixed_destination").replace_strict(
                old_zones,
                new_clusters,
                default=None,
                return_dtype=pl.UInt32,
            )
        )
        .sort(["context_id", "layer"])
    )
    selected_initial = (
        initial_locations.filter(pl.col("context_id").is_in(context_ids))
        .with_columns(
            initial_zone=pl.col("initial_zone").replace_strict(
                old_zones,
                new_clusters,
                return_dtype=pl.UInt32,
            )
        )
        .sort("context_id")
    )
    if od_costs["origin"].n_unique() != n_clusters:
        raise ValueError("coarse OD table does not contain every cluster")
    return od_costs, destination_inputs, selected_steps, selected_initial


def solve_contexts_rust(
    *,
    context_ids: list[int],
    steps: pl.DataFrame,
    initial_locations: pl.DataFrame,
    fine: FineInputs,
    coarse: CoarseInputs,
    wrapped_home_shadow_price: float,
    bidirectional_feasibility: bool,
    n_threads: int | None,
) -> tuple[dict[int, ContextResult], dict[str, object]]:
    (
        od_costs,
        destination_inputs,
        selected_steps,
        selected_initial,
    ) = rust_tables(
        context_ids=context_ids,
        steps=steps,
        initial_locations=initial_locations,
        fine=fine,
        coarse=coarse,
    )
    sampler = DestinationSampler(
        od_costs=od_costs,
        destination_inputs=destination_inputs,
    )
    output = sampler.solve_second_order(
        steps=selected_steps,
        initial_locations=selected_initial,
        logit_scale=LOGIT_SCALE,
        update_plan_timings=True,
        use_shadow_prices=True,
        wrapped_home_time_shadow_price=wrapped_home_shadow_price,
        use_bidirectional_feasibility=bidirectional_feasibility,
        n_threads=n_threads,
        skip_infeasible=True,
    )
    output_contexts = np.asarray(output["context_ids"], dtype=np.uint64)
    log_partitions = np.asarray(output["log_partitions"])
    probabilities = np.asarray(
        output["first_destination_probabilities"]
    ).reshape(output_contexts.size, coarse.costs.shape[0])
    zone_ids = np.asarray(output["zone_ids"])
    if not np.array_equal(
        zone_ids,
        np.arange(coarse.costs.shape[0], dtype=zone_ids.dtype),
    ):
        raise ValueError("Rust returned unexpected coarse zone ordering")
    return (
        {
            int(context_id): ContextResult(
                float(log_partition),
                probability,
            )
            for context_id, log_partition, probability in zip(
                output_contexts,
                log_partitions,
                probabilities,
                strict=True,
            )
        },
        output,
    )


def selected_contexts(
    steps: pl.DataFrame,
    n_contexts: int,
    seed: int,
) -> tuple[list[int], dict[int, list[dict[str, float | int | None]]]]:
    supported = (
        steps.group_by("context_id")
        .agg(
            anchors=pl.col("anchor_id").drop_nulls().n_unique(),
            layers=pl.len(),
        )
        .filter(pl.col("anchors") <= 1)
        .with_columns(
            sample_order=pl.struct(
                ["layers", "context_id"]
            ).hash(seed=seed)
        )
        .sort("sample_order")
        .head(n_contexts)
    )
    context_ids = supported["context_id"].to_list()
    selected_steps = (
        steps.filter(pl.col("context_id").is_in(context_ids))
        .sort(["context_id", "layer"])
    )
    by_context = {
        int(partition["context_id"][0]): partition.drop(
            ["context_id", "layer"]
        ).to_dicts()
        for partition in selected_steps.partition_by(
            "context_id",
            maintain_order=True,
        )
    }
    return [int(value) for value in context_ids], by_context


def coarse_context_count(
    steps: pl.DataFrame,
    initial_locations: pl.DataFrame,
    zone_ids: np.ndarray,
    labels: np.ndarray,
) -> int:
    zone_to_cluster = {
        int(zone): int(cluster)
        for zone, cluster in zip(zone_ids, labels, strict=True)
    }
    old = list(zone_to_cluster)
    new = list(zone_to_cluster.values())
    fixed_cluster = pl.col("fixed_destination").replace_strict(
        old,
        new,
        default=None,
        return_dtype=pl.UInt32,
    )
    step_columns = [
        "layer",
        "activity_id",
        "anchor_id",
        "fixed_cluster",
        "departure_time",
        "arrival_time",
        "arrival_time_rigidity",
        "next_departure_time",
        "duration_per_person",
        "value_of_time",
        "mean_duration_per_person",
        "min_activity_time",
    ]
    profiles = (
        steps.with_columns(fixed_cluster=fixed_cluster)
        .with_columns(step_hash=pl.struct(step_columns).hash(seed=71))
        .group_by("context_id")
        .agg(
            sequence_hash=pl.col("step_hash")
            .sort_by("layer")
            .cast(pl.String)
            .str.join("-")
        )
    )
    homes = initial_locations.with_columns(
        initial_cluster=pl.col("initial_zone").replace_strict(
            old,
            new,
            return_dtype=pl.UInt32,
        )
    )
    return (
        profiles.join(
            homes.select(["context_id", "initial_cluster"]),
            on="context_id",
        )
        .select(["sequence_hash", "initial_cluster"])
        .unique()
        .height
    )


def lift_probability(
    cluster_probability: np.ndarray,
    *,
    first_activity: int,
    fine: FineInputs,
    coarse: CoarseInputs,
) -> np.ndarray:
    result = np.zeros(fine.zone_ids.size)
    for cluster, probability in enumerate(cluster_probability):
        if probability <= 0.0:
            continue
        members = np.flatnonzero(coarse.labels == cluster)
        capacity = fine.capacity[first_activity, members]
        if capacity.sum() > 0.0:
            result[members] = probability * capacity / capacity.sum()
        else:
            result[members] = probability / members.size
    return result


def main() -> None:
    args = parse_args()
    files = resolve_snapshot_files(args.group_day_trips_folder)
    preparation_started = time.perf_counter()
    od_costs = prepare_od_costs(
        files["transport_costs"],
        files["demand_groups"],
    )
    destination_inputs = prepare_destination_inputs(
        files["destination_saturation"],
        files["demand_groups"],
    )
    steps, initial_locations, _ = prepare_complete_contexts(
        activity_sequences_path=files["activity_sequences"],
        survey_plan_steps_path=files["survey_plan_steps"],
        demand_groups_path=files["demand_groups"],
        activity_dur_path=files["activity_dur"],
    )
    fine = dense_inputs(od_costs, destination_inputs)
    kernel, embedding = travel_kernel_embedding(
        fine.costs,
        args.embedding_rank,
    )
    context_ids, steps_by_context = selected_contexts(
        steps,
        args.n_contexts,
        args.seed,
    )
    initial_by_context = {
        int(context_id): int(initial_zone)
        for context_id, initial_zone in initial_locations.filter(
            pl.col("context_id").is_in(context_ids)
        ).iter_rows()
    }
    zone_index = {
        int(zone): index for index, zone in enumerate(fine.zone_ids)
    }
    activity_index = {
        int(activity): index
        for index, activity in enumerate(fine.activity_ids)
    }
    preparation_seconds = time.perf_counter() - preparation_started

    supported_count = (
        steps.group_by("context_id")
        .agg(anchors=pl.col("anchor_id").drop_nulls().n_unique())
        .filter(pl.col("anchors") <= 1)
        .height
    )
    print("Aggregated exact backward-forward experiment")
    print(
        f"zones={fine.zone_ids.size:,}; contexts tested={len(context_ids):,}; "
        f"supported contexts={supported_count:,}/{steps['context_id'].n_unique():,}"
    )
    print(
        f"kernel embedding rank={embedding.shape[1] // 2}; "
        f"input preparation={preparation_seconds:.2f}s"
    )
    print()

    lifted_by_resolution: dict[int, dict[int, np.ndarray]] = {}
    log_partition_by_resolution: dict[int, dict[int, float]] = {}
    for n_clusters in args.clusters:
        aggregation_started = time.perf_counter()
        if n_clusters == fine.zone_ids.size:
            # Full resolution is an identity hierarchy. Avoid running k-means
            # and matrix aggregation when the requested clusters are the raw
            # transport zones themselves.
            labels = np.arange(fine.zone_ids.size, dtype=np.intp)
            coarse = CoarseInputs(
                labels=labels,
                costs=fine.costs,
                times=fine.times,
                capacity=fine.capacity,
                country_coefficient=fine.country_coefficient,
                shadow_price=fine.shadow_price,
            )
        else:
            labels = cluster_zones(
                embedding,
                n_clusters,
                args.seed,
            )
            coarse = aggregate_inputs(fine, kernel, labels)
        aggregate_seconds = time.perf_counter() - aggregation_started
        unique_contexts = coarse_context_count(
            steps,
            initial_locations,
            fine.zone_ids,
            labels,
        )
        zone_to_cluster = {
            int(zone): int(cluster)
            for zone, cluster in zip(
                fine.zone_ids,
                labels,
                strict=True,
            )
        }

        solve_started = time.perf_counter()
        results, rust_report = solve_contexts_rust(
            context_ids=context_ids,
            steps=steps,
            initial_locations=initial_locations,
            fine=fine,
            coarse=coarse,
            wrapped_home_shadow_price=args.wrapped_home_shadow_price,
            bidirectional_feasibility=args.bidirectional_feasibility,
            n_threads=args.n_threads,
        )
        lifted: dict[int, np.ndarray] = {}
        log_partitions: dict[int, float] = {}
        infeasible = 0
        for context_id in context_ids:
            steps_for_context = steps_by_context[context_id]
            result = results.get(context_id)
            if result is None or not math.isfinite(result.log_partition):
                infeasible += 1
                continue
            first_activity = activity_index[
                int(steps_for_context[0]["activity_id"])
            ]
            lifted[context_id] = lift_probability(
                result.first_destination_probability,
                first_activity=first_activity,
                fine=fine,
                coarse=coarse,
            )
            log_partitions[context_id] = result.log_partition
        solve_seconds = time.perf_counter() - solve_started
        lifted_by_resolution[n_clusters] = lifted
        log_partition_by_resolution[n_clusters] = log_partitions
        common_log_partitions = list(log_partitions.values())
        rust_metrics = ""
        if rust_report is not None:
            duration_checks = int(rust_report["duration_checks"])
            duration_infeasible = int(
                rust_report["duration_infeasible"]
            )
            pair_states = int(rust_report["pair_states"])
            feasible_pair_states = int(
                rust_report["feasible_pair_states"]
            )
            forward_pair_states = int(rust_report["forward_pair_states"])
            forward_reachable_pair_states = int(
                rust_report["forward_reachable_pair_states"]
            )
            corridor_pair_states = int(rust_report["corridor_pair_states"])
            forward_time_edge_scans = int(
                rust_report["forward_time_edge_scans"]
            )
            forward_time_cutoffs = int(rust_report["forward_time_cutoffs"])
            backward_time_edge_scans = int(
                rust_report["backward_time_edge_scans"]
            )
            backward_time_cutoffs = int(
                rust_report["backward_time_cutoffs"])
            corridor_metrics = ""
            if forward_pair_states:
                corridor_metrics = (
                    f"forward-reachable="
                    f"{forward_reachable_pair_states / forward_pair_states:.1%}; "
                    f"corridor="
                    f"{corridor_pair_states / max(pair_states, 1):.1%}; "
                    f"time-scans/cutoffs="
                    f"{forward_time_edge_scans + backward_time_edge_scans:,}/"
                    f"{forward_time_cutoffs + backward_time_cutoffs:,}; "
                )
            rust_metrics = (
                f"Rust core={float(rust_report['wall_seconds']):.2f}s; "
                f"duration-pruned="
                f"{duration_infeasible / max(duration_checks, 1):.1%}; "
                f"dead pair states="
                f"{1.0 - feasible_pair_states / max(pair_states, 1):.1%}; "
                f"{corridor_metrics}"
            )
        print(
            f"{n_clusters:3d} clusters: "
            f"size min/median/max="
            f"{np.bincount(labels).min()}/"
            f"{np.median(np.bincount(labels)):.0f}/"
            f"{np.bincount(labels).max()}; "
            f"coarse contexts={unique_contexts:,}; "
            f"aggregate={aggregate_seconds:.2f}s; "
            f"solve={solve_seconds:.2f}s "
            f"({len(context_ids) / solve_seconds:.1f} contexts/s); "
            f"{rust_metrics}"
            f"infeasible={infeasible}; "
            f"logZ median={np.median(common_log_partitions):.3f}"
        )

    print()
    print("Resolution convergence for the first destination")
    for previous, current in zip(
        args.clusters[:-1],
        args.clusters[1:],
        strict=True,
    ):
        common = sorted(
            set(lifted_by_resolution[previous])
            & set(lifted_by_resolution[current])
        )
        total_variations = [
            0.5
            * np.abs(
                lifted_by_resolution[previous][context]
                - lifted_by_resolution[current][context]
            ).sum()
            for context in common
        ]
        log_partition_changes = [
            abs(
                log_partition_by_resolution[previous][context]
                - log_partition_by_resolution[current][context]
            )
            for context in common
        ]
        print(
            f"{previous:3d} -> {current:3d}: "
            f"TV p50/p95={np.median(total_variations):.3f}/"
            f"{np.quantile(total_variations, 0.95):.3f}; "
            f"|delta logZ| p50/p95="
            f"{np.median(log_partition_changes):.3f}/"
            f"{np.quantile(log_partition_changes, 0.95):.3f}"
        )


if __name__ == "__main__":
    main()
