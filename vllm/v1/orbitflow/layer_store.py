# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import os
from pathlib import Path

import torch

from vllm.v1.orbitflow.profiler import (
    OrbitFlowProfile,
    OrbitFlowRuntimeProfiler,
)


class OrbitFlowLayerStore:
    """Pinned-CPU backing for layer-reused GPU staging blocks."""

    def __init__(
        self,
        layer_names: tuple[str, ...],
        num_resident_layers: int,
        num_gpu_blocks: int,
        validate_transfers: bool = False,
        cpu_cache_bytes: int | None = None,
        nvme_path: str | None = None,
        nvme_bytes: int = 0,
        coalesce_transfers: bool = True,
    ):
        self._layer_names = layer_names
        self._layer_indices = {name: i for i, name in enumerate(layer_names)}
        self._num_resident = num_resident_layers
        self._num_gpu_blocks = num_gpu_blocks
        self._validate_transfers = validate_transfers
        self._cpu_cache_bytes = cpu_cache_bytes
        self._nvme_path = Path(nvme_path).expanduser() if nvme_path else None
        self._nvme_bytes = nvme_bytes
        self._coalesce_transfers = coalesce_transfers
        if self._cpu_cache_bytes is not None and self._cpu_cache_bytes <= 0:
            raise ValueError("cpu_cache_bytes must be positive")
        if self._nvme_bytes < 0:
            raise ValueError("nvme_bytes must be non-negative")
        if self._nvme_path is not None:
            if self._nvme_bytes <= 0:
                raise ValueError("nvme_bytes must be positive with nvme_path")
            self._nvme_path.mkdir(parents=True, exist_ok=True)
        self._kv_caches: dict[str, torch.Tensor] = {}
        self._request_blocks: dict[str, tuple[tuple[int, ...], ...]] = {}
        self._active_request_ids: tuple[str, ...] = ()
        self._resident_layers: dict[str, frozenset[int]] = {}
        self._deposit_layers: dict[str, frozenset[int]] = {}
        self._completed_deposit_request_ids: set[str] = set()
        self._force_load_layers: dict[str, frozenset[int]] = {}
        self._cpu: dict[tuple[str, str], torch.Tensor] = {}
        self._nvme: dict[
            tuple[str, str], tuple[Path, tuple[int, ...], torch.dtype, int]
        ] = {}
        self._last_access: dict[tuple[str, str], int] = {}
        self._access_clock = 0
        self._stream: torch.cuda.Stream | None = None
        self._load_events: dict[str, torch.cuda.Event] = {}
        self._validated_layers: set[str] = set()
        self._profiler = OrbitFlowRuntimeProfiler()

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
        topology_changed = bool(finished_request_ids)
        for req_id, groups in request_blocks.items():
            previous = self._request_blocks.get(req_id)
            if previous is None or any(
                len(old) != len(new)
                for old, new in zip(previous, groups, strict=True)
            ):
                topology_changed = True
                break
        # A previous D2H may still target an arena that must now be resized or
        # released. Fence only topology changes; steady-state decode remains
        # fully asynchronous across scheduler steps.
        if topology_changed and self._stream is not None:
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
        for layer_idx in range(len(self._layer_names)):
            owners: dict[int, str] = {}
            for req_id, groups in request_blocks.items():
                for gpu_id in groups[layer_idx]:
                    previous_owner = owners.setdefault(gpu_id, req_id)
                    if previous_owner != req_id:
                        raise RuntimeError(
                            "OrbitFlow staging alias within a layer: "
                            f"layer={layer_idx}, gpu_block={gpu_id}, "
                            f"requests={previous_owner},{req_id}"
                        )
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
                num_blocks = len(groups[layer_idx])
                key = (req_id, layer_name)
                current = self._cpu.get(key)
                if current is None and key in self._nvme:
                    current = self._get_cpu(key)
                if current is None or current.shape[0] < num_blocks:
                    block = self._block_view(kv_layer, 0)
                    new_bytes = (
                        num_blocks * block.numel() * block.element_size()
                    )
                    self._make_cpu_room(
                        max(
                            new_bytes
                            - (current.nbytes if current is not None else 0),
                            0,
                        ),
                        exclude={key},
                    )
                    arena = torch.zeros(
                        (num_blocks, *block.shape),
                        dtype=block.dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                    if current is not None:
                        arena[: current.shape[0]].copy_(current)
                    self._cpu[key] = arena
                    self._touch(key)

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
            start = self._profiler.start_transfer(stream)
            num_bytes = 0
            for req_id in self._active_request_ids:
                if not self._needs_load(req_id, layer_idx):
                    continue
                groups = self._request_blocks[req_id]
                gpu_ids = groups[layer_idx]
                src = self._get_cpu((req_id, layer_name))
                gpu = self._blocks_view(kv_layer)
                for logical_idx, gpu_id, count in self._transfer_runs(gpu_ids):
                    source = src.narrow(0, logical_idx, count)
                    gpu.narrow(0, gpu_id, count).copy_(
                        source, non_blocking=True
                    )
                    num_bytes += source.nbytes
            self._profiler.end_transfer(start, stream, num_bytes, "h2d")
            event = torch.cuda.Event()
            event.record(stream)
        self._load_events[layer_name] = event

    def wait_for_layer(self, layer_name: str) -> None:
        layer_idx = self._layer_indices[layer_name]
        compute_stream = torch.cuda.current_stream(
            self._kv_caches[layer_name].device
        )
        if not self._has_offloaded_request(layer_idx):
            self._profiler.record_compute_start(layer_name, compute_stream)
            return
        if layer_name not in self._load_events:
            self.start_load_layer(layer_name)
        event = self._load_events[layer_name]
        compute_stream.wait_event(event)
        self._profiler.record_compute_start(layer_name, compute_stream)
        if self._validate_transfers and layer_name not in self._validated_layers:
            self._validate_layer(layer_name, layer_idx)
            self._validated_layers.add(layer_name)

    def store_layer(self, layer_name: str) -> None:
        layer_idx = self._layer_indices[layer_name]
        kv_layer = self._kv_caches[layer_name]
        compute_stream = torch.cuda.current_stream(kv_layer.device)
        self._profiler.record_compute_end(layer_name, compute_stream)
        if not self._has_offloaded_request(layer_idx):
            return
        stream = self._require_stream()
        stream.wait_stream(compute_stream)
        with torch.cuda.stream(stream):
            start = self._profiler.start_transfer(stream)
            num_bytes = 0
            for req_id in self._active_request_ids:
                if layer_idx in self._resident_layers.get(
                    req_id, ()
                ) and layer_idx not in self._deposit_layers.get(req_id, ()):
                    continue
                groups = self._request_blocks[req_id]
                gpu_ids = groups[layer_idx]
                dst = self._get_cpu((req_id, layer_name))
                gpu = self._blocks_view(kv_layer)
                for logical_idx, gpu_id, count in self._transfer_runs(gpu_ids):
                    target = dst.narrow(0, logical_idx, count)
                    target.copy_(
                        gpu.narrow(0, gpu_id, count), non_blocking=True
                    )
                    num_bytes += target.nbytes
            self._profiler.end_transfer(start, stream, num_bytes, "d2h")

    def release(self, request_id: str) -> None:
        self._request_blocks.pop(request_id, None)
        self._resident_layers.pop(request_id, None)
        self._deposit_layers.pop(request_id, None)
        self._force_load_layers.pop(request_id, None)
        for key in [key for key in self._cpu if key[0] == request_id]:
            del self._cpu[key]
            self._last_access.pop(key, None)
        for key in [key for key in self._nvme if key[0] == request_id]:
            path, _, _, _ = self._nvme.pop(key)
            path.unlink(missing_ok=True)
            self._last_access.pop(key, None)

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

    def wait_for_pending_saves(self) -> None:
        if self._stream is not None:
            self._stream.synchronize()
        self._completed_deposit_request_ids.update(self._deposit_layers)

    def take_completed_deposit_request_ids(self) -> tuple[str, ...]:
        completed = tuple(sorted(self._completed_deposit_request_ids))
        self._completed_deposit_request_ids.clear()
        return completed

    def _validate_layer(self, layer_name: str, layer_idx: int) -> None:
        kv_layer = self._kv_caches[layer_name]
        for req_id in self._active_request_ids:
            if not self._needs_load(req_id, layer_idx):
                continue
            groups = self._request_blocks[req_id]
            expected_arena = self._get_cpu((req_id, layer_name))
            for logical_idx, gpu_id in enumerate(groups[layer_idx]):
                expected = expected_arena[logical_idx]
                actual = self._blocks_view(kv_layer)[gpu_id]
                if not torch.equal(actual.cpu(), expected):
                    raise RuntimeError(
                        "OrbitFlow transfer validation failed for "
                        f"request={req_id}, layer={layer_name}, "
                        f"logical_block={logical_idx}, gpu_block={gpu_id}"
                    )

    def _block_view(self, kv_layer: torch.Tensor, block_id: int) -> torch.Tensor:
        return self._blocks_view(kv_layer)[block_id]

    def _blocks_view(self, kv_layer: torch.Tensor) -> torch.Tensor:
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
        return kv_layer.movedim(block_dims[0], 0)

    @staticmethod
    def _contiguous_runs(
        gpu_ids: tuple[int, ...],
    ) -> list[tuple[int, int, int]]:
        if not gpu_ids:
            return []
        runs = []
        logical_start = 0
        gpu_start = gpu_ids[0]
        for logical_idx in range(1, len(gpu_ids)):
            if gpu_ids[logical_idx] != gpu_ids[logical_idx - 1] + 1:
                runs.append(
                    (logical_start, gpu_start, logical_idx - logical_start)
                )
                logical_start = logical_idx
                gpu_start = gpu_ids[logical_idx]
        runs.append((logical_start, gpu_start, len(gpu_ids) - logical_start))
        return runs

    def collect_profile(self) -> OrbitFlowProfile:
        return self._profiler.collect()

    def _transfer_runs(
        self, gpu_ids: tuple[int, ...]
    ) -> list[tuple[int, int, int]]:
        if self._coalesce_transfers:
            return self._contiguous_runs(gpu_ids)
        return [(index, gpu_id, 1) for index, gpu_id in enumerate(gpu_ids)]

    @property
    def cpu_bytes(self) -> int:
        return sum(tensor.nbytes for tensor in self._cpu.values())

    @property
    def nvme_used_bytes(self) -> int:
        return sum(metadata[3] for metadata in self._nvme.values())

    def _touch(self, key: tuple[str, str]) -> None:
        self._access_clock += 1
        self._last_access[key] = self._access_clock

    def _get_cpu(self, key: tuple[str, str]) -> torch.Tensor:
        tensor = self._cpu.get(key)
        if tensor is not None:
            self._touch(key)
            return tensor
        path, shape, dtype, num_bytes = self._nvme[key]
        self._make_cpu_room(num_bytes, exclude={key})
        tensor = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
        mapped = torch.from_file(
            str(path), shared=False, size=tensor.numel(), dtype=dtype
        ).view(shape)
        tensor.copy_(mapped)
        self._cpu[key] = tensor
        self._touch(key)
        return tensor

    def _make_cpu_room(
        self, needed: int, *, exclude: set[tuple[str, str]]
    ) -> None:
        if self._cpu_cache_bytes is None:
            return
        while self.cpu_bytes + needed > self._cpu_cache_bytes:
            candidates = [
                key for key in self._cpu
                if key not in exclude
            ]
            if not candidates:
                if needed > self._cpu_cache_bytes and self._nvme_path is not None:
                    return
                raise MemoryError(
                    "OrbitFlow pinned CPU cache budget is too small for an arena"
                )
            victim = min(candidates, key=lambda key: self._last_access.get(key, 0))
            self._spill(victim)

    def _spill(self, key: tuple[str, str]) -> None:
        if self._nvme_path is None:
            raise MemoryError(
                "OrbitFlow pinned CPU cache is full and NVMe tier is disabled"
            )
        if self._stream is not None:
            self._stream.synchronize()
        tensor = self._cpu[key]
        num_bytes = tensor.nbytes
        existing = self._nvme.get(key)
        existing_bytes = existing[3] if existing is not None else 0
        if self.nvme_used_bytes - existing_bytes + num_bytes > self._nvme_bytes:
            raise MemoryError("OrbitFlow NVMe cache budget exceeded")
        digest = hashlib.sha256(
            f"{key[0]}\0{key[1]}".encode()
        ).hexdigest()
        path = self._nvme_path / f"{digest}.kv"
        with path.open("wb") as file:
            file.truncate(num_bytes)
        mapped = torch.from_file(
            str(path),
            shared=True,
            size=tensor.numel(),
            dtype=tensor.dtype,
        ).view(tensor.shape)
        mapped.copy_(tensor)
        if hasattr(os, "sync"):
            del mapped
        self._nvme[key] = (path, tuple(tensor.shape), tensor.dtype, num_bytes)
        del self._cpu[key]
