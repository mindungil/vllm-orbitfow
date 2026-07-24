# SPDX-License-Identifier: Apache-2.0

from vllm.v1.orbitflow.runtime import (
    LayerTransferAction,
    OrbitFlowLayerPipeline,
)
from vllm.v1.orbitflow.types import (
    BatchPlacement,
    PlacementReason,
    RequestPlacement,
)


def _placement() -> BatchPlacement:
    return BatchPlacement(
        epoch=1,
        created_at_step=0,
        expires_at_step=10,
        reason=PlacementReason.INITIAL,
        requests=(
            RequestPlacement("a", (0, 2), 1.0, 10, False),
            RequestPlacement("b", (1,), 1.0, 10, False),
        ),
    )


def test_pipeline_prefetches_and_waits_by_layer():
    loads: list[LayerTransferAction] = []
    waits: list[LayerTransferAction] = []
    stores: list[LayerTransferAction] = []
    pipeline = OrbitFlowLayerPipeline(
        ("l0", "l1", "l2"),
        prefetch_distance=1,
        load=loads.append,
        wait=waits.append,
        store=stores.append,
    )

    pipeline.begin_batch(_placement())
    assert loads == [
        LayerTransferAction(0, ("b",)),
        LayerTransferAction(1, ("a",)),
    ]

    pipeline.wait_for_layer("l0")
    pipeline.store_layer("l0")
    pipeline.wait_for_layer("l1")
    pipeline.wait_for_layer("l2")

    assert loads == [
        LayerTransferAction(0, ("b",)),
        LayerTransferAction(1, ("a",)),
        LayerTransferAction(2, ("b",)),
    ]
    assert waits == [
        LayerTransferAction(0, ("b",)),
        LayerTransferAction(1, ("a",)),
        LayerTransferAction(2, ("b",)),
    ]
    assert stores == [LayerTransferAction(0, ("b",))]


def test_pipeline_resets_between_batches():
    loads: list[LayerTransferAction] = []
    pipeline = OrbitFlowLayerPipeline(
        ("l0", "l1", "l2"),
        prefetch_distance=0,
        load=loads.append,
        wait=lambda _: None,
        store=lambda _: None,
    )
    placement = _placement()

    pipeline.begin_batch(placement)
    pipeline.wait_for_layer("l2")
    pipeline.begin_batch(placement)

    assert loads.count(LayerTransferAction(0, ("b",))) == 2
