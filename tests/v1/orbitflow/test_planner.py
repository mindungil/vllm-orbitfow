# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.v1.orbitflow import (
    BatchPlacement,
    OrbitFlowBlockAllocator,
    OrbitFlowConfig,
    OrbitFlowController,
    OrbitFlowPlanner,
    PlacementReason,
    RequestPlacement,
    RequestProfile,
)
from vllm.v1.orbitflow.scheduler import OrbitFlowConnectorScheduler


def profile(
    request_id: str,
    *,
    kv_bytes: int = 100,
    transfer_ms: float = 2.0,
    slo_ms: float = 5.0,
) -> RequestProfile:
    return RequestProfile(
        request_id=request_id,
        kv_bytes_per_layer=kv_bytes,
        compute_ms=1.0,
        transfer_ms_per_layer=transfer_ms,
        tbt_slo_ms=slo_ms,
    )


def test_planner_uses_request_wise_placements() -> None:
    planner = OrbitFlowPlanner(OrbitFlowConfig(num_layers=4, gpu_capacity_bytes=500))

    placement = planner.plan(
        [
            profile("expensive", transfer_ms=3.0),
            profile("cheap", transfer_ms=0.5),
        ],
        epoch=1,
        step=0,
        reason=PlacementReason.INITIAL,
    )

    assert placement.gpu_bytes <= 500
    assert placement.for_request("expensive").num_gpu_layers == 4
    assert placement.for_request("cheap").num_gpu_layers == 0
    assert not placement.paused_request_ids


def test_planner_pauses_largest_request_when_infeasible() -> None:
    planner = OrbitFlowPlanner(OrbitFlowConfig(num_layers=4, gpu_capacity_bytes=300))

    placement = planner.plan(
        [profile("large", kv_bytes=200), profile("small", kv_bytes=100)],
        epoch=1,
        step=0,
        reason=PlacementReason.INITIAL,
    )

    assert placement.paused_request_ids == ("large",)
    assert tuple(item.request_id for item in placement.requests) == ("small",)


def test_token_deposit_can_make_placement_feasible() -> None:
    planner = OrbitFlowPlanner(OrbitFlowConfig(num_layers=4, gpu_capacity_bytes=0))
    deposited = RequestProfile(
        request_id="request",
        kv_bytes_per_layer=100,
        compute_ms=1,
        transfer_ms_per_layer=2,
        tbt_slo_ms=5,
        deposit_ms=5,
    )

    placement = planner.plan(
        [deposited],
        epoch=1,
        step=0,
        reason=PlacementReason.INITIAL,
    )

    assert placement.for_request("request").gpu_layers == ()
    assert not placement.for_request("request").violates_slo


def test_planner_never_pauses_last_request_for_slo_only() -> None:
    planner = OrbitFlowPlanner(
        OrbitFlowConfig(num_layers=4, gpu_capacity_bytes=0)
    )

    placement = planner.plan(
        [profile("request", transfer_ms=100, slo_ms=1)],
        epoch=1,
        step=0,
        reason=PlacementReason.INITIAL,
    )

    assert not placement.paused_request_ids
    assert placement.for_request("request").violates_slo


def test_controller_replans_on_batch_change_and_profile_mismatch() -> None:
    controller = OrbitFlowController(
        OrbitFlowConfig(
            num_layers=4,
            gpu_capacity_bytes=800,
            profile_mismatch_ratio=0.1,
        )
    )
    initial = controller.update([profile("a")], step=0)
    same = controller.update([profile("a")], step=0, actual_tbt_ms=100)
    changed = controller.update([profile("a"), profile("b")], step=1)
    mismatched = controller.update(
        [profile("a"), profile("b")],
        step=2,
        actual_tbt_ms=100,
    )

    assert same is initial
    assert changed.epoch == 2
    assert changed.reason is PlacementReason.BATCH_CHANGED
    assert mismatched.epoch == 3
    assert mismatched.reason is PlacementReason.PROFILE_MISMATCH


def test_promotion_only_migration_updates_same_step_placement() -> None:
    scheduler = OrbitFlowConnectorScheduler.__new__(
        OrbitFlowConnectorScheduler
    )
    target = RequestPlacement("request", (0, 1), 1.0, 200, False)
    scheduler._controller = Mock(
        update=Mock(
            return_value=BatchPlacement(
                epoch=2,
                created_at_step=1,
                expires_at_step=2,
                reason=PlacementReason.BATCH_CHANGED,
                requests=(target,),
            )
        )
    )
    scheduler._num_layers = 2
    scheduler._block_size = 16
    scheduler._page_size_bytes = 100
    scheduler._transfer_bandwidth = 12.0
    scheduler._compute_ms = 0.1
    scheduler._default_tbt_slo_ms = 100.0
    scheduler._measured_bandwidth_bytes_per_ms = None
    scheduler._measured_layer_ms = None
    scheduler._last_actual_tbt_ms = None
    scheduler._requests = {}
    scheduler._locked_gpu_layers = {"request": (0,)}
    scheduler._pending_migrations = {}
    scheduler._ready_migrations = {}
    scheduler._prepared_placement = None
    scheduler._physical_placement = Mock(side_effect=lambda placement: placement)
    scheduler._enforce_physical_capacity = Mock(
        side_effect=lambda placement: placement
    )
    request = SimpleNamespace(
        request_id="request",
        num_tokens=16,
        kv_transfer_params=None,
    )

    placement = scheduler.prepare_requests([request], step=1)

    assert placement.for_request("request").gpu_layers == (0, 1)
    assert scheduler.take_ready_migrations() == {
        "request": ((0,), (0, 1))
    }


def test_config_rejects_invalid_capacity() -> None:
    with pytest.raises(ValueError, match="gpu_capacity_bytes"):
        OrbitFlowConfig(num_layers=4, gpu_capacity_bytes=-1)


def test_block_allocator_reuses_staging_across_layers() -> None:
    planner = OrbitFlowPlanner(OrbitFlowConfig(num_layers=2, gpu_capacity_bytes=20))
    placement = planner.plan(
        [
            RequestProfile("a", 10, 1, 1, 2),
            RequestProfile("b", 10, 1, 0.1, 2),
        ],
        epoch=1,
        step=0,
        reason=PlacementReason.INITIAL,
    )
    allocator = OrbitFlowBlockAllocator(
        num_gpu_blocks=8,
        num_staging_blocks=4,
    )

    block_plan = allocator.build_plan(
        placement,
        {"a": 2, "b": 2},
        num_layers=2,
    )

    assert block_plan.placement_epoch == placement.epoch
    offloaded_layers = [
        layer
        for layer in block_plan.layers
        if any(request.load for request in layer.requests)
    ]
    assert offloaded_layers
    for layer in offloaded_layers:
        staging_ids = [
            block_id
            for request in layer.requests
            for block_id in request.gpu_block_ids
            if request.load
        ]
        assert set(staging_ids) <= {4, 5, 6, 7}


def test_block_allocator_rejects_staging_overflow() -> None:
    planner = OrbitFlowPlanner(OrbitFlowConfig(num_layers=2, gpu_capacity_bytes=0))
    placement = planner.plan(
        [
            RequestProfile(
                "request",
                kv_bytes_per_layer=10,
                compute_ms=1,
                transfer_ms_per_layer=1,
                tbt_slo_ms=10,
            )
        ],
        epoch=1,
        step=0,
        reason=PlacementReason.INITIAL,
    )
    allocator = OrbitFlowBlockAllocator(4, 2)

    with pytest.raises(MemoryError, match="staging blocks"):
        allocator.build_plan(placement, {"request": 3}, num_layers=2)
