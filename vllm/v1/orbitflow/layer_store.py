# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


class OrbitFlowLayerStore:
    """Pinned-CPU backing for layer-reused GPU staging blocks."""

    def __init__(
        self,
        layer_names: tuple[str, ...],
        num_resident_layers: int,
        num_gpu_blocks: int,
        validate_transfers: bool = False,
    ):
        self._layer_names = layer_names
        self._layer_indices = {name: i for i, name in enumerate(layer_names)}
        self._num_resident = num_resident_layers
        self._num_gpu_blocks = num_gpu_blocks
        self._validate_transfers = validate_transfers
        self._kv_caches: dict[str, torch.Tensor] = {}
        self._request_blocks: dict[str, tuple[tuple[int, ...], ...]] = {}
        self._active_request_ids: tuple[str, ...] = ()
        self._resident_layers: dict[str, frozenset[int]] = {}
        self._deposit_layers: dict[str, frozenset[int]] = {}
        self._force_load_layers: dict[str, frozenset[int]] = {}
        self._cpu: dict[tuple[str, str, int], torch.Tensor] = {}
        self._stream: torch.cuda.Stream | None = None
        self._load_events: dict[str, torch.cuda.Event] = {}
        self._validated_layers: set[str] = set()

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._kv_caches = kv_caches
        device = next(iter(kv_caches.values())).device
        self._stream = torch.cuda.Stream(device=device)

    def begin_batch(
        self,
        request_blocks: dict[str, tuple[tuple[int, ...], ...]],
        finished_request_ids: set[str],
        resident_layers: dict[str, frozenset[int]],
        migration_deposit_layers: dict[str, tuple[int, ...]],
    ) -> None:
        if finished_request_ids and self._stream is not None:
            self._stream.synchronize()
        for req_id in finished_request_ids:
            self.release(req_id)
        old_blocks = dict(self._request_blocks)
        old_residence = dict(self._resident_layers)
        self._request_blocks.update(request_blocks)
        self._resident_layers.update(resident_layers)
        self._deposit_layers = {
            req_id: frozenset(layers)
            for req_id, layers in migration_deposit_layers.items()
        }
        self._force_load_layers = {}
        for req_id, groups in request_blocks.items():
            previous_groups = old_blocks.get(req_id)
            previous_resident = old_residence.get(req_id, frozenset())
            forced = {
                layer
                for layer, group in enumerate(groups)
                if (
                    previous_groups is not None
                    and tuple(previous_groups[layer]) != tuple(group)
                    and layer in resident_layers.get(req_id, ())
                    and layer not in previous_resident
                )
            }
            if forced:
                self._force_load_layers[req_id] = frozenset(forced)
        self._active_request_ids = tuple(request_blocks)
        self._load_events.clear()
        self._validated_layers.clear()
        for req_id, groups in request_blocks.items():
            request_resident = self._resident_layers.get(
                req_id, frozenset(range(self._num_resident))
            )
            for layer_idx in range(len(self._layer_names)):
                if (
                    layer_idx in request_resident
                    and layer_idx not in self._deposit_layers.get(req_id, ())
                ):
                    continue
                layer_name = self._layer_names[layer_idx]
                kv_layer = self._kv_caches[layer_name]
                for logical_idx in range(len(groups[layer_idx])):
                    key = (req_id, layer_name, logical_idx)
                    if key not in self._cpu:
                        self._cpu[key] = torch.zeros_like(
                            self._block_view(kv_layer, 0),
                            device="cpu",
                            pin_memory=True,
                        )

    def start_load_layer(self, layer_name: str) -> None:
        layer_idx = self._layer_indices[layer_name]
        if (
            not self._has_offloaded_request(layer_idx)
            or layer_name in self._load_events
        ):
            return
        stream = self._require_stream()
        kv_layer = self._kv_caches[layer_name]
        with torch.cuda.stream(stream):
            for req_id in self._active_request_ids:
                if not self._needs_load(req_id, layer_idx):
                    continue
                groups = self._request_blocks[req_id]
                for logical_idx, gpu_id in enumerate(groups[layer_idx]):
                    src = self._cpu[(req_id, layer_name, logical_idx)]
                    self._block_view(kv_layer, gpu_id).copy_(src, non_blocking=True)
            event = torch.cuda.Event()
            event.record(stream)
        self._load_events[layer_name] = event

    def wait_for_layer(self, layer_name: str) -> None:
        layer_idx = self._layer_indices[layer_name]
        if not self._has_offloaded_request(layer_idx):
            return
        if layer_name not in self._load_events:
            self.start_load_layer(layer_name)
        event = self._load_events[layer_name]
        torch.cuda.current_stream(self._kv_caches[layer_name].device).wait_event(event)
        if self._validate_transfers and layer_name not in self._validated_layers:
            self._validate_layer(layer_name, layer_idx)
            self._validated_layers.add(layer_name)

    def store_layer(self, layer_name: str) -> None:
        layer_idx = self._layer_indices[layer_name]
        if not self._has_offloaded_request(layer_idx):
            return
        stream = self._require_stream()
        kv_layer = self._kv_caches[layer_name]
        stream.wait_stream(torch.cuda.current_stream(kv_layer.device))
        with torch.cuda.stream(stream):
            for req_id in self._active_request_ids:
                if layer_idx in self._resident_layers.get(
                    req_id, ()
                ) and layer_idx not in self._deposit_layers.get(req_id, ()):
                    continue
                groups = self._request_blocks[req_id]
                for logical_idx, gpu_id in enumerate(groups[layer_idx]):
                    dst = self._cpu[(req_id, layer_name, logical_idx)]
                    dst.copy_(self._block_view(kv_layer, gpu_id), non_blocking=True)

    def release(self, request_id: str) -> None:
        self._request_blocks.pop(request_id, None)
        self._resident_layers.pop(request_id, None)
        self._deposit_layers.pop(request_id, None)
        self._force_load_layers.pop(request_id, None)
        for key in [key for key in self._cpu if key[0] == request_id]:
            del self._cpu[key]

    def _require_stream(self) -> torch.cuda.Stream:
        if self._stream is None:
            raise RuntimeError("KV caches must be registered before execution")
        return self._stream

    def _has_offloaded_request(self, layer_idx: int) -> bool:
        return any(
            self._needs_load(req_id, layer_idx) for req_id in self._active_request_ids
        )

    def _needs_load(self, request_id: str, layer_idx: int) -> bool:
        return layer_idx not in self._resident_layers.get(
            request_id, ()
        ) or layer_idx in self._force_load_layers.get(request_id, ())

    def wait_for_pending_deposits(self) -> None:
        if self._deposit_layers and self._stream is not None:
            self._stream.synchronize()

    def _validate_layer(self, layer_name: str, layer_idx: int) -> None:
        kv_layer = self._kv_caches[layer_name]
        for req_id in self._active_request_ids:
            if not self._needs_load(req_id, layer_idx):
                continue
            groups = self._request_blocks[req_id]
            for logical_idx, gpu_id in enumerate(groups[layer_idx]):
                expected = self._cpu[(req_id, layer_name, logical_idx)]
                actual = self._block_view(kv_layer, gpu_id)
                if not torch.equal(actual.cpu(), expected):
                    raise RuntimeError(
                        "OrbitFlow transfer validation failed for "
                        f"request={req_id}, layer={layer_name}, "
                        f"logical_block={logical_idx}, gpu_block={gpu_id}"
                    )

    def _block_view(self, kv_layer: torch.Tensor, block_id: int) -> torch.Tensor:
        block_dims = [
            dim
            for dim, size in enumerate(kv_layer.shape)
            if size == self._num_gpu_blocks
        ]
        if len(block_dims) != 1:
            raise RuntimeError(
                "cannot identify KV cache block dimension from shape "
                f"{tuple(kv_layer.shape)} and num_blocks={self._num_gpu_blocks}"
            )
        return kv_layer.select(block_dims[0], block_id)
