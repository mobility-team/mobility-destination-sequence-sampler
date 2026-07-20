from __future__ import annotations

import argparse
import sqlite3
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy.linalg import svd

from mobility_destination_sequence_sampler._core import (
    benchmark_hierarchical_kernel,
)
from experiments.benchmarks.perf_grand_geneve_cache import (
    DEFAULT_GROUP_DAY_TRIPS_FOLDER,
    LOGIT_SCALE,
    prepare_complete_contexts,
    prepare_destination_inputs,
    prepare_od_costs,
    resolve_snapshot_files,
)


DEFAULT_TRANSPORT_ZONES = Path(
    r"D:\data\mobility\projects\grand-geneve"
    r"\b8f7a3c54bdeef48b24ee5ecd4dc1518-transport_zones.gpkg"
)


@dataclass(frozen=True)
class SpatialBlock:
    left: np.ndarray
    right: np.ndarray
    separation_ratio: float
    is_far: bool


@dataclass(frozen=True)
class SpatialCluster:
    indices: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    children: tuple[SpatialCluster, SpatialCluster] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the numerical rank of the Grand Genève destination "
            "travel kernel without running destination sampling."
        )
    )
    parser.add_argument(
        "--group-day-trips-folder",
        type=Path,
        default=DEFAULT_GROUP_DAY_TRIPS_FOLDER,
    )
    parser.add_argument(
        "--transport-zones",
        type=Path,
        default=DEFAULT_TRANSPORT_ZONES,
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[0.1, LOGIT_SCALE, 0.5, 1.0],
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64, 128],
    )
    parser.add_argument("--leaf-size", type=int, default=64)
    parser.add_argument(
        "--far-separation",
        type=float,
        default=1.0,
        help=(
            "A leaf pair is far when bounding-box distance divided by the "
            "larger box diameter reaches this value."
        ),
    )
    parser.add_argument(
        "--benchmark-right-hand-sides",
        type=int,
        nargs="+",
        default=[1, 64, 512],
    )
    parser.add_argument("--benchmark-repetitions", type=int, default=7)
    parser.add_argument(
        "--hierarchical-benchmark-energy",
        type=float,
        default=0.99,
    )
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def dense_od_matrices(
    od_costs: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zone_ids = np.unique(
        np.concatenate(
            [
                od_costs["origin"].to_numpy(),
                od_costs["destination"].to_numpy(),
            ]
        )
    )
    zone_index = {int(zone_id): index for index, zone_id in enumerate(zone_ids)}
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
    costs = np.full((zone_ids.size, zone_ids.size), np.inf, dtype=np.float64)
    times = np.full((zone_ids.size, zone_ids.size), np.inf, dtype=np.float64)
    costs[origins, destinations] = od_costs["cost"].to_numpy()
    times[origins, destinations] = od_costs["time"].to_numpy()
    np.fill_diagonal(costs, 0.0)
    np.fill_diagonal(times, 0.0)
    return zone_ids.astype(np.uint32, copy=False), costs, times


def read_zone_coordinates(path: Path, zone_ids: np.ndarray) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Transport-zone file does not exist: {path}")

    with sqlite3.connect(path) as connection:
        feature_tables = [
            row[0]
            for row in connection.execute(
                "SELECT table_name FROM gpkg_contents "
                "WHERE data_type = 'features'"
            )
        ]
        if len(feature_tables) != 1:
            raise ValueError(
                f"Expected one feature table in {path}, found {feature_tables}"
            )
        table = feature_tables[0].replace('"', '""')
        rows = connection.execute(
            f'SELECT transport_zone_id, x, y FROM "{table}"'
        ).fetchall()

    coordinate_by_zone = {
        int(zone_id): (float(x), float(y)) for zone_id, x, y in rows
    }
    missing = [
        int(zone_id)
        for zone_id in zone_ids
        if int(zone_id) not in coordinate_by_zone
    ]
    if missing:
        raise ValueError(
            f"{len(missing)} OD zones have no coordinates; first IDs: "
            f"{missing[:10]}"
        )
    return np.asarray(
        [coordinate_by_zone[int(zone_id)] for zone_id in zone_ids],
        dtype=np.float64,
    )


def effective_rank(
    singular_values: np.ndarray,
    retained_energy: float,
) -> int:
    energy = np.square(singular_values)
    threshold = retained_energy * energy.sum()
    return int(np.searchsorted(np.cumsum(energy), threshold) + 1)


def build_spatial_cluster(
    coordinates: np.ndarray,
    leaf_size: int,
) -> SpatialCluster:
    def build(indices: np.ndarray) -> SpatialCluster:
        points = coordinates[indices]
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        if indices.size <= leaf_size:
            return SpatialCluster(indices, minimum, maximum, None)

        axis = int(np.argmax(maximum - minimum))
        order = indices[np.argsort(points[:, axis], kind="stable")]
        middle = order.size // 2
        return SpatialCluster(
            indices,
            minimum,
            maximum,
            (build(order[:middle]), build(order[middle:])),
        )

    return build(np.arange(coordinates.shape[0], dtype=np.intp))


def bounding_box_distance(
    left: SpatialCluster,
    right: SpatialCluster,
) -> tuple[float, float]:
    gap = np.maximum(
        np.maximum(
            left.minimum - right.maximum,
            right.minimum - left.maximum,
        ),
        0.0,
    )
    distance = float(np.linalg.norm(gap))
    largest_diameter = max(
        float(np.linalg.norm(left.maximum - left.minimum)),
        float(np.linalg.norm(right.maximum - right.minimum)),
        1.0,
    )
    return distance, largest_diameter


def spatial_blocks(
    root: SpatialCluster,
    far_separation: float,
) -> list[SpatialBlock]:
    blocks: list[SpatialBlock] = []

    def partition(left: SpatialCluster, right: SpatialCluster) -> None:
        distance, diameter = bounding_box_distance(left, right)
        separation_ratio = distance / diameter
        if separation_ratio >= far_separation:
            blocks.append(
                SpatialBlock(
                    left=left.indices,
                    right=right.indices,
                    separation_ratio=separation_ratio,
                    is_far=True,
                )
            )
            return

        if left.children is None and right.children is None:
            blocks.append(
                SpatialBlock(
                    left=left.indices,
                    right=right.indices,
                    separation_ratio=separation_ratio,
                    is_far=False,
                )
            )
            return

        left_diameter = float(np.linalg.norm(left.maximum - left.minimum))
        right_diameter = float(np.linalg.norm(right.maximum - right.minimum))
        if right.children is None or (
            left.children is not None and left_diameter >= right_diameter
        ):
            assert left.children is not None
            partition(left.children[0], right)
            partition(left.children[1], right)
        else:
            assert right.children is not None
            partition(left, right.children[0])
            partition(left, right.children[1])

    partition(root, root)
    return blocks


def summarize_block_ranks(
    kernel: np.ndarray,
    blocks: list[SpatialBlock],
    test_vectors: np.ndarray,
    seed: int,
) -> None:
    far_blocks = [block for block in blocks if block.is_far]
    near_blocks = len(blocks) - len(far_blocks)
    print()
    print("Adaptive hierarchical matrix at the model scale")
    print(f"matrix blocks: {len(blocks):,}")
    print(
        f"low-rank far blocks: {len(far_blocks):,}; "
        f"exact near blocks: {near_blocks:,}"
    )
    if not far_blocks:
        print("No blocks meet the requested far-separation threshold.")
        return

    decompositions = []
    decomposition_started = time.perf_counter()
    for spatial_block in far_blocks:
        matrix_block = kernel[
            np.ix_(spatial_block.left, spatial_block.right)
        ]
        decompositions.append(
            svd(
                matrix_block,
                full_matrices=False,
                check_finite=False,
                overwrite_a=False,
            )
        )
    decomposition_elapsed = time.perf_counter() - decomposition_started
    dense_storage = sum(
        block.left.size * block.right.size
        for block in blocks
        if not block.is_far
    )
    sampled_origins = np.random.default_rng(seed).choice(
        kernel.shape[0],
        size=min(128, kernel.shape[0]),
        replace=False,
    )
    exact_messages = kernel @ test_vectors

    for retained_energy in [0.99, 0.999]:
        ranks = []
        approximate_kernel = kernel.copy()
        compressed_storage = dense_storage
        started = time.perf_counter()
        for spatial_block, decomposition in zip(
            far_blocks,
            decompositions,
            strict=True,
        ):
            left, singular_values, right_transpose = decomposition
            rank = effective_rank(singular_values, retained_energy)
            ranks.append(rank)
            compressed_storage += rank * (
                spatial_block.left.size
                + spatial_block.right.size
                + 1
            )
            approximate_kernel[
                np.ix_(spatial_block.left, spatial_block.right)
            ] = (
                left[:, :rank] * singular_values[:rank]
            ) @ right_transpose[:rank]

        elapsed = time.perf_counter() - started
        ranks.sort()
        p90_index = min(int(0.9 * len(ranks)), len(ranks) - 1)
        compression = kernel.size / compressed_storage
        kernel_error = (
            np.linalg.norm(approximate_kernel - kernel)
            / np.linalg.norm(kernel)
        )
        approximate_messages = approximate_kernel @ test_vectors
        message_errors = np.linalg.norm(
            approximate_messages - exact_messages,
            axis=0,
        ) / np.maximum(np.linalg.norm(exact_messages, axis=0), 1e-15)
        clipped_kernel = np.maximum(approximate_kernel, 0.0)
        negative_mass = (
            np.abs(approximate_kernel[approximate_kernel < 0.0]).sum()
            / kernel.sum()
        )
        total_variations = []
        for origin in sampled_origins:
            for vector in test_vectors.T:
                exact_weights = kernel[origin] * vector
                approximate_weights = clipped_kernel[origin] * vector
                exact_total = exact_weights.sum()
                approximate_total = approximate_weights.sum()
                if exact_total == 0.0 or approximate_total == 0.0:
                    continue
                total_variations.append(
                    0.5
                    * np.abs(
                        exact_weights / exact_total
                        - approximate_weights / approximate_total
                    ).sum()
                )
        print(
            f"{retained_energy * 100:5.1f}% energy: "
            f"rank median={statistics.median(ranks):.1f}, "
            f"p90={ranks[p90_index]}, max={max(ranks)}, "
            f"whole storage={compression:.2f}x, "
            f"kernel error={kernel_error:.5f}, "
            f"message p50/max={np.median(message_errors):.5f}/"
            f"{message_errors.max():.5f}, "
            f"TV p50/p95={np.median(total_variations):.5f}/"
            f"{np.quantile(total_variations, 0.95):.5f}, "
            f"negative mass={negative_mass:.3g}"
        )
        print(f"       reconstruction and validation: {elapsed:.3f} s")
    print(f"far-block decompositions: {decomposition_elapsed:.3f} s")


def encode_hierarchical_blocks(
    kernel: np.ndarray,
    blocks: list[SpatialBlock],
    retained_energy: float,
) -> list[
    tuple[bool, list[int], list[int], int, list[float], list[float]]
]:
    encoded = []
    for block in blocks:
        matrix_block = kernel[np.ix_(block.left, block.right)]
        if block.is_far:
            left, singular_values, right_transpose = svd(
                matrix_block,
                full_matrices=False,
                check_finite=False,
                overwrite_a=False,
            )
            rank = effective_rank(singular_values, retained_energy)
            left_factor = left[:, :rank] * singular_values[:rank]
            encoded.append(
                (
                    True,
                    block.left.tolist(),
                    block.right.tolist(),
                    rank,
                    left_factor.ravel().tolist(),
                    right_transpose[:rank].ravel().tolist(),
                )
            )
        else:
            encoded.append(
                (
                    False,
                    block.left.tolist(),
                    block.right.tolist(),
                    0,
                    matrix_block.ravel().tolist(),
                    [],
                )
            )
    return encoded


def benchmark_hierarchical_products(
    kernel: np.ndarray,
    blocks: list[SpatialBlock],
    retained_energy: float,
    right_hand_sides: list[int],
    repetitions: int,
) -> None:
    encoding_started = time.perf_counter()
    encoded_blocks = encode_hierarchical_blocks(
        kernel,
        blocks,
        retained_energy,
    )
    encoding_elapsed = time.perf_counter() - encoding_started
    dense_kernel = kernel.ravel().tolist()
    print()
    print(
        "Rust hierarchical products "
        f"({retained_energy * 100:g}% energy per far block)"
    )
    print(f"factor encoding: {encoding_elapsed:.3f} s")
    print("rhs  dense ms  hierarchy ms  speedup  maximum error")
    for n_rhs in right_hand_sides:
        report = benchmark_hierarchical_kernel(
            n_zones=kernel.shape[0],
            dense_kernel=dense_kernel,
            blocks=encoded_blocks,
            n_right_hand_sides=n_rhs,
            repetitions=repetitions,
        )
        print(
            f"{n_rhs:3d}  "
            f"{report['dense_seconds'] * 1_000:8.3f}  "
            f"{report['hierarchical_seconds'] * 1_000:12.3f}  "
            f"{report['speedup']:7.2f}x  "
            f"{report['maximum_absolute_error']:.6g}"
        )


def realistic_test_vectors(
    destination_inputs: pl.DataFrame,
    zone_ids: np.ndarray,
    *,
    n_random: int,
    seed: int,
) -> np.ndarray:
    zone_index = {int(zone_id): index for index, zone_id in enumerate(zone_ids)}
    vectors = []
    for activity_id in destination_inputs["activity_id"].unique().sort():
        activity = destination_inputs.filter(
            pl.col("activity_id") == activity_id
        )
        values = np.zeros(zone_ids.size, dtype=np.float64)
        for row in activity.iter_rows(named=True):
            destination_index = zone_index.get(int(row["destination"]))
            if destination_index is None:
                continue
            exponent = LOGIT_SCALE * (
                float(row["saturation_utility"])
                - float(row["shadow_price"])
            )
            values[destination_index] = max(
                float(row["opportunity_capacity"]),
                0.0,
            ) * np.exp(np.clip(exponent, -50.0, 50.0))
        if values.sum() > 0.0:
            values /= values.sum()
            vectors.append(values)

    rng = np.random.default_rng(seed)
    base_vectors = list(vectors)
    for _ in range(n_random):
        base = base_vectors[int(rng.integers(len(base_vectors)))]
        continuation = np.exp(rng.normal(0.0, 1.0, size=zone_ids.size))
        vector = base * continuation
        vector /= vector.sum()
        vectors.append(vector)
    return np.column_stack(vectors)


def approximation_report(
    kernel: np.ndarray,
    left: np.ndarray,
    singular_values: np.ndarray,
    right_transpose: np.ndarray,
    test_vectors: np.ndarray,
    ranks: list[int],
    seed: int,
) -> None:
    exact_messages = kernel @ test_vectors
    origin_rng = np.random.default_rng(seed)
    sampled_origins = origin_rng.choice(
        kernel.shape[0],
        size=min(128, kernel.shape[0]),
        replace=False,
    )

    print()
    print("Global truncated-SVD approximation at the model scale")
    print(
        "rank  kernel error  message p50  message max  "
        "conditional TV p50/p95  negative mass"
    )
    kernel_norm = np.linalg.norm(kernel)
    for rank in ranks:
        if rank > singular_values.size:
            continue
        factors_left = left[:, :rank] * singular_values[:rank]
        factors_right = right_transpose[:rank]
        approximate_messages = factors_left @ (
            factors_right @ test_vectors
        )
        message_errors = np.linalg.norm(
            approximate_messages - exact_messages,
            axis=0,
        ) / np.maximum(np.linalg.norm(exact_messages, axis=0), 1e-15)

        approximate_kernel = factors_left @ factors_right
        negative_mass = (
            np.abs(approximate_kernel[approximate_kernel < 0.0]).sum()
            / kernel.sum()
        )
        clipped_kernel = np.maximum(approximate_kernel, 0.0)
        total_variations = []
        for origin in sampled_origins:
            for vector in test_vectors.T:
                exact_weights = kernel[origin] * vector
                approximate_weights = clipped_kernel[origin] * vector
                exact_total = exact_weights.sum()
                approximate_total = approximate_weights.sum()
                if exact_total == 0.0 or approximate_total == 0.0:
                    continue
                total_variations.append(
                    0.5
                    * np.abs(
                        exact_weights / exact_total
                        - approximate_weights / approximate_total
                    ).sum()
                )

        kernel_error = np.sqrt(
            np.square(singular_values[rank:]).sum()
        ) / kernel_norm
        print(
            f"{rank:4d}  {kernel_error:12.5f}  "
            f"{np.median(message_errors):11.5f}  "
            f"{message_errors.max():11.5f}  "
            f"{np.median(total_variations):7.5f}/"
            f"{np.quantile(total_variations, 0.95):7.5f}  "
            f"{negative_mass:13.6g}"
        )


def median_runtime(
    operation,
    *,
    repetitions: int,
) -> float:
    operation()
    timings = []
    for _ in range(repetitions):
        started = time.perf_counter()
        operation()
        timings.append(time.perf_counter() - started)
    return statistics.median(timings)


def benchmark_products(
    kernel: np.ndarray,
    left: np.ndarray,
    singular_values: np.ndarray,
    right_transpose: np.ndarray,
    ranks: list[int],
    right_hand_sides: list[int],
    repetitions: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    print()
    print("Dense versus factored matrix products")
    print("rhs  rank  dense ms  factored ms  speedup")
    for n_rhs in right_hand_sides:
        vectors = rng.random((kernel.shape[1], n_rhs))
        dense_seconds = median_runtime(
            lambda: kernel @ vectors,
            repetitions=repetitions,
        )
        for rank in ranks:
            if rank > singular_values.size:
                continue
            factors_left = left[:, :rank] * singular_values[:rank]
            factors_right = right_transpose[:rank]
            factored_seconds = median_runtime(
                lambda: factors_left @ (factors_right @ vectors),
                repetitions=repetitions,
            )
            print(
                f"{n_rhs:3d}  {rank:4d}  "
                f"{dense_seconds * 1_000:8.3f}  "
                f"{factored_seconds * 1_000:11.3f}  "
                f"{dense_seconds / factored_seconds:7.2f}x"
        )


def representative_steps(steps: pl.DataFrame) -> pl.DataFrame:
    return (
        steps.filter(
            pl.col("fixed_destination").is_null()
            & (pl.col("activity_id") != 0)
        )
        .with_columns(
            available_window=(
                pl.col("next_departure_time") - pl.col("departure_time")
            )
        )
        .sort(["activity_id", "available_window"])
        .group_by("activity_id", maintain_order=True)
        .agg(pl.all().get(pl.len() // 2))
        .sort("activity_id")
    )


def report_step_profile_counts(steps: pl.DataFrame) -> None:
    variable_steps = steps.filter(
        pl.col("fixed_destination").is_null()
        & (pl.col("activity_id") != 0)
    ).with_columns(
        available_window=(
            pl.col("next_departure_time") - pl.col("departure_time")
        )
    )
    profile_columns = [
        "activity_id",
        "available_window",
        "mean_duration_per_person",
        "min_activity_time",
        "value_of_time",
    ]
    print()
    print("Transition parameter reuse")
    print(f"variable activity steps: {variable_steps.height:,}")
    print(
        "exact transition profiles: "
        f"{variable_steps.select(profile_columns).unique().height:,}"
    )
    print(
        "activity-parameter profiles without time window: "
        f"{variable_steps.select(profile_columns[:1] + profile_columns[2:]).unique().height:,}"
    )
    print(
        "distinct activity time windows: "
        f"{variable_steps['available_window'].n_unique():,}"
    )
    for minutes in [5, 10, 15, 30, 60]:
        width = minutes / 60.0
        binned = variable_steps.with_columns(
            window_bin=(
                pl.col("available_window") / width
            ).round()
            * width
        )
        binned_profiles = binned.select(
            [
                "activity_id",
                "window_bin",
                "mean_duration_per_person",
                "min_activity_time",
                "value_of_time",
            ]
        ).unique()
        print(
            f"{minutes:2d}-minute window profiles: "
            f"{binned_profiles.height:,}"
        )


def activity_transition_kernel(
    *,
    step: dict[str, float | int],
    destination_inputs: pl.DataFrame,
    zone_ids: np.ndarray,
    costs: np.ndarray,
    times: np.ndarray,
    scale: float,
) -> np.ndarray:
    zone_index = {int(zone_id): index for index, zone_id in enumerate(zone_ids)}
    log_capacity = np.full(zone_ids.size, -np.inf, dtype=np.float64)
    coefficient = np.zeros(zone_ids.size, dtype=np.float64)
    activity_inputs = destination_inputs.filter(
        pl.col("activity_id") == int(step["activity_id"])
    )
    for row in activity_inputs.iter_rows(named=True):
        destination = zone_index.get(int(row["destination"]))
        if destination is None or float(row["opportunity_capacity"]) <= 0.0:
            continue
        log_capacity[destination] = np.log(
            float(row["opportunity_capacity"])
        )
        coefficient[destination] = (
            float(row["country_value_coefficient"])
            * float(step["value_of_time"])
            + float(row["shadow_price"])
        )

    available_duration = float(step["available_window"]) - times
    duration_factor = np.zeros_like(times)
    positive_duration = available_duration > 0.0
    duration_factor[positive_duration] = np.maximum(
        np.log(
            available_duration[positive_duration]
            / float(step["min_activity_time"])
        ),
        0.0,
    )
    activity_utility = (
        coefficient[np.newaxis, :]
        * float(step["mean_duration_per_person"])
        * duration_factor
    )
    log_kernel = (
        log_capacity[np.newaxis, :]
        + scale * (activity_utility - costs)
    )
    log_kernel[
        ~np.isfinite(costs)
        | ~positive_duration
        | ~np.isfinite(log_capacity)[np.newaxis, :]
    ] = -np.inf
    finite = np.isfinite(log_kernel)
    if not finite.any():
        return np.zeros_like(costs)
    log_kernel[finite] -= log_kernel[finite].max()
    kernel = np.zeros_like(costs)
    kernel[finite] = np.exp(np.maximum(log_kernel[finite], -745.0))
    return kernel


def compact_hierarchy_metrics(
    kernel: np.ndarray,
    blocks: list[SpatialBlock],
    retained_energy: float,
) -> tuple[float, float, float, int, int]:
    approximate_kernel = kernel.copy()
    storage = 0
    ranks = []
    for block in blocks:
        matrix_block = kernel[np.ix_(block.left, block.right)]
        if block.is_far:
            left, singular_values, right_transpose = svd(
                matrix_block,
                full_matrices=False,
                check_finite=False,
                overwrite_a=False,
            )
            rank = effective_rank(singular_values, retained_energy)
            ranks.append(rank)
            storage += rank * (
                block.left.size + block.right.size + 1
            )
            approximate_kernel[
                np.ix_(block.left, block.right)
            ] = (
                left[:, :rank] * singular_values[:rank]
            ) @ right_transpose[:rank]
        else:
            storage += matrix_block.size

    kernel_norm = np.linalg.norm(kernel)
    error = (
        np.linalg.norm(approximate_kernel - kernel) / kernel_norm
        if kernel_norm > 0.0
        else 0.0
    )
    ranks.sort()
    p90_index = min(int(0.9 * len(ranks)), len(ranks) - 1)
    return (
        kernel.size / storage,
        error,
        statistics.median(ranks),
        ranks[p90_index],
        max(ranks),
    )


def report_real_transition_ranks(
    *,
    steps: pl.DataFrame,
    destination_inputs: pl.DataFrame,
    zone_ids: np.ndarray,
    costs: np.ndarray,
    times: np.ndarray,
    blocks: list[SpatialBlock],
) -> None:
    print()
    print("Representative complete transition factors")
    print(
        "activity  window  global r99/r999  hierarchy storage/error  "
        "far rank median/p90/max"
    )
    for step in representative_steps(steps).iter_rows(named=True):
        kernel = activity_transition_kernel(
            step=step,
            destination_inputs=destination_inputs,
            zone_ids=zone_ids,
            costs=costs,
            times=times,
            scale=LOGIT_SCALE,
        )
        singular_values = svd(
            kernel,
            compute_uv=False,
            check_finite=False,
            overwrite_a=False,
        )
        storage, error, median_rank, p90_rank, maximum_rank = (
            compact_hierarchy_metrics(kernel, blocks, 0.99)
        )
        print(
            f"{int(step['activity_id']):8d}  "
            f"{float(step['available_window']):6.2f}  "
            f"{effective_rank(singular_values, 0.99):4d}/"
            f"{effective_rank(singular_values, 0.999):4d}       "
            f"{storage:5.2f}x/{error:7.5f}       "
            f"{median_rank:4.1f}/{p90_rank:2d}/{maximum_rank:2d}"
        )


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
    steps, _, _ = prepare_complete_contexts(
        activity_sequences_path=files["activity_sequences"],
        survey_plan_steps_path=files["survey_plan_steps"],
        demand_groups_path=files["demand_groups"],
        activity_dur_path=files["activity_dur"],
    )
    zone_ids, costs, times = dense_od_matrices(od_costs)
    coordinates = read_zone_coordinates(args.transport_zones, zone_ids)
    preparation_elapsed = time.perf_counter() - preparation_started

    finite_edges = np.isfinite(costs).sum()
    print("Grand Genève travel-kernel rank exploration")
    print(f"zones: {zone_ids.size:,}")
    print(
        f"finite OD pairs: {finite_edges:,}/"
        f"{costs.size:,} ({finite_edges / costs.size:.2%})"
    )
    print(
        f"finite cost range: "
        f"{costs[np.isfinite(costs)].min():.3f} to "
        f"{costs[np.isfinite(costs)].max():.3f}"
    )
    print(f"input preparation: {preparation_elapsed:.3f} s")

    model_decomposition = None
    model_kernel = None
    for scale in args.scales:
        kernel = np.exp(-scale * costs)
        decomposition_started = time.perf_counter()
        left, singular_values, right_transpose = svd(
            kernel,
            full_matrices=False,
            check_finite=False,
            overwrite_a=False,
        )
        decomposition_elapsed = time.perf_counter() - decomposition_started
        print()
        print(f"scale={scale:g}")
        print(
            "global energy ranks: "
            f"90%={effective_rank(singular_values, 0.90)}, "
            f"99%={effective_rank(singular_values, 0.99)}, "
            f"99.9%={effective_rank(singular_values, 0.999)}, "
            f"99.99%={effective_rank(singular_values, 0.9999)}"
        )
        print(
            f"leading singular-value ratio s1/s2="
            f"{singular_values[0] / singular_values[1]:.3f}; "
            f"full SVD={decomposition_elapsed:.3f} s"
        )
        if np.isclose(scale, LOGIT_SCALE):
            model_kernel = kernel
            model_decomposition = (
                left,
                singular_values,
                right_transpose,
            )

    if model_kernel is None or model_decomposition is None:
        raise ValueError(
            f"--scales must contain the model scale {LOGIT_SCALE}"
        )
    left, singular_values, right_transpose = model_decomposition
    test_vectors = realistic_test_vectors(
        destination_inputs,
        zone_ids,
        n_random=16,
        seed=args.seed,
    )
    approximation_report(
        model_kernel,
        left,
        singular_values,
        right_transpose,
        test_vectors,
        args.ranks,
        args.seed,
    )
    benchmark_products(
        model_kernel,
        left,
        singular_values,
        right_transpose,
        args.ranks,
        args.benchmark_right_hand_sides,
        args.benchmark_repetitions,
        args.seed,
    )

    root = build_spatial_cluster(coordinates, args.leaf_size)
    blocks = spatial_blocks(root, args.far_separation)
    summarize_block_ranks(
        model_kernel,
        blocks,
        test_vectors,
        args.seed,
    )
    benchmark_hierarchical_products(
        model_kernel,
        blocks,
        args.hierarchical_benchmark_energy,
        args.benchmark_right_hand_sides,
        args.benchmark_repetitions,
    )
    report_real_transition_ranks(
        steps=steps,
        destination_inputs=destination_inputs,
        zone_ids=zone_ids,
        costs=costs,
        times=times,
        blocks=blocks,
    )
    report_step_profile_counts(steps)


if __name__ == "__main__":
    main()
