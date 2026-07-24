# SPDX-License-Identifier: Apache-2.0

from vllm.v1.orbitflow.metadata import OrbitFlowWorkerMetadata


def test_worker_profile_aggregation_preserves_per_rank_bandwidth() -> None:
    first = OrbitFlowWorkerMetadata(
        compute_ms=4,
        compute_layers=2,
        h2d_ms=3,
        h2d_bytes=100,
        completed_deposit_request_ids=("a", "b"),
    )
    second = OrbitFlowWorkerMetadata(
        compute_ms=6,
        compute_layers=2,
        h2d_ms=5,
        h2d_bytes=100,
        completed_deposit_request_ids=("b", "c"),
    )

    result = first.aggregate(second)

    assert isinstance(result, OrbitFlowWorkerMetadata)
    assert result.compute_ms == 10
    assert result.compute_layers == 4
    assert result.h2d_ms == 5
    assert result.h2d_bytes == 100
    assert result.completed_deposit_request_ids == ("b",)
