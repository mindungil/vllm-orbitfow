# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest
import torch

from vllm.v1.orbitflow.layer_store import OrbitFlowLayerStore


def test_transfer_validation_runs_once_per_layer_and_batch() -> None:
    store = OrbitFlowLayerStore(("layer.0",), 0, 1, validate_transfers=True)
    store._has_offloaded_request = Mock(return_value=True)
    store.start_load_layer = Mock()
    store._load_events["layer.0"] = Mock()
    store._kv_caches["layer.0"] = Mock(device="cuda")
    store._validate_layer = Mock()
    store._profiler = Mock()

    with patch("torch.cuda.current_stream", return_value=Mock()):
        store.wait_for_layer("layer.0")
        store.wait_for_layer("layer.0")

    store._validate_layer.assert_called_once_with("layer.0", 0)


def test_contiguous_runs_coalesce_adjacent_blocks() -> None:
    assert OrbitFlowLayerStore._contiguous_runs((5, 6, 7, 12, 13, 2)) == [
        (0, 5, 3),
        (3, 12, 2),
        (5, 2, 1),
    ]


def test_deposit_completion_is_reported_once() -> None:
    store = OrbitFlowLayerStore(("layer.0",), 0, 1)
    store._deposit_layers = {"request": frozenset({0})}

    store.wait_for_pending_saves()

    assert store.take_completed_deposit_request_ids() == ("request",)
    assert store.take_completed_deposit_request_ids() == ()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires pinned memory")
def test_nvme_spill_and_reload(tmp_path) -> None:
    store = OrbitFlowLayerStore(
        ("layer.0",),
        0,
        1,
        cpu_cache_bytes=16,
        nvme_path=str(tmp_path),
        nvme_bytes=64,
    )
    first = ("request", "layer.0")
    second = ("request", "layer.1")
    store._cpu[first] = torch.arange(4, dtype=torch.float32)
    store._last_access[first] = 1

    store._make_cpu_room(16, exclude={second})

    assert first not in store._cpu
    assert first in store._nvme
    assert torch.equal(
        store._get_cpu(first), torch.arange(4, dtype=torch.float32)
    )
