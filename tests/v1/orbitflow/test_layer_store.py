# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

from vllm.v1.orbitflow.layer_store import OrbitFlowLayerStore


def test_transfer_validation_runs_once_per_layer_and_batch() -> None:
    store = OrbitFlowLayerStore(("layer.0",), 0, 1, validate_transfers=True)
    store._has_offloaded_request = Mock(return_value=True)
    store.start_load_layer = Mock()
    store._load_events["layer.0"] = Mock()
    store._kv_caches["layer.0"] = Mock(device="cuda")
    store._validate_layer = Mock()

    with patch("torch.cuda.current_stream", return_value=Mock()):
        store.wait_for_layer("layer.0")
        store.wait_for_layer("layer.0")

    store._validate_layer.assert_called_once_with("layer.0", 0)
