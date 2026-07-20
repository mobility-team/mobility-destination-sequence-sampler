from __future__ import annotations

from mobility_destination_sequence_sampler._core import (
    benchmark_hierarchical_kernel,
)


def test_hierarchical_product_matches_dense_product() -> None:
    report = benchmark_hierarchical_kernel(
        n_zones=2,
        dense_kernel=[1.0, 2.0, 3.0, 4.0],
        blocks=[
            (
                False,
                [0, 1],
                [0, 1],
                0,
                [1.0, 2.0, 3.0, 4.0],
                [],
            )
        ],
        n_right_hand_sides=4,
        repetitions=1,
    )

    assert report["blocks"] == 1
    assert report["maximum_absolute_error"] == 0.0
    assert report["checksum_difference"] == 0.0
    assert report["dense_seconds"] > 0.0
    assert report["hierarchical_seconds"] > 0.0
