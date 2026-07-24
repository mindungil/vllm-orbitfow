# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

from vllm.v1.orbitflow.types import BatchPlacement


@dataclass(frozen=True, slots=True)
class BlockTransfer:
    cpu_block_id: int
    gpu_block_id: int


@dataclass(frozen=True, slots=True)
class RequestLayerBlocks:
    request_id: str
    gpu_block_ids: tuple[int, ...]
    load: tuple[BlockTransfer, ...]
    store: tuple[BlockTransfer, ...]


@dataclass(frozen=True, slots=True)
class LayerBlockPlan:
    layer_index: int
    requests: tuple[RequestLayerBlocks, ...]


@dataclass(frozen=True, slots=True)
class BatchBlockPlan:
    placement_epoch: int
    layers: tuple[LayerBlockPlan, ...]


class OrbitFlowBlockAllocator:
    """Maps logical layer blocks to resident or reusable staging GPU slots."""

    def __init__(self, num_gpu_blocks: int, num_staging_blocks: int):
        if num_gpu_blocks <= 0:
            raise ValueError("num_gpu_blocks must be positive")
        if not 0 < num_staging_blocks < num_gpu_blocks:
            raise ValueError(
                "num_staging_blocks must be between 1 and num_gpu_blocks - 1"
            )
        resident_end = num_gpu_blocks - num_staging_blocks
        self._free_resident = list(range(resident_end - 1, -1, -1))
        self._staging = tuple(range(resident_end, num_gpu_blocks))
        self._resident: dict[tuple[str, int, int], int] = {}
        self._cpu_blocks: dict[tuple[str, int, int], int] = {}
        self._next_cpu_block = 0

    @property
    def num_free_resident_blocks(self) -> int:
        return len(self._free_resident)

    def release_request(self, request_id: str) -> None:
        resident_keys = [key for key in self._resident if key[0] == request_id]
        for key in resident_keys:
            self._free_resident.append(self._resident.pop(key))
        cpu_keys = [key for key in self._cpu_blocks if key[0] == request_id]
        for key in cpu_keys:
            del self._cpu_blocks[key]

    def build_plan(
        self,
        placement: BatchPlacement,
        block_counts: dict[str, int],
        *,
        num_layers: int,
    ) -> BatchBlockPlan:
        placements = {request.request_id: request for request in placement.requests}
        if missing := placements.keys() - block_counts.keys():
            raise KeyError(f"missing block counts for requests: {sorted(missing)}")

        desired_resident = {
            (request_id, layer, block_index)
            for request_id, request in placements.items()
            for layer in request.gpu_layers
            for block_index in range(block_counts[request_id])
        }
        self._reconcile_resident(desired_resident)

        layer_plans = []
        for layer in range(num_layers):
            staging_cursor = 0
            request_plans = []
            for request_id, request in placements.items():
                count = block_counts[request_id]
                keys = [
                    (request_id, layer, block_index) for block_index in range(count)
                ]
                if layer in request.gpu_layers:
                    gpu_ids = tuple(self._resident[key] for key in keys)
                    request_plans.append(
                        RequestLayerBlocks(request_id, gpu_ids, (), ())
                    )
                    continue

                staging_end = staging_cursor + count
                if staging_end > len(self._staging):
                    raise MemoryError(
                        f"layer {layer} needs {staging_end} staging blocks, "
                        f"only {len(self._staging)} are reserved"
                    )
                gpu_ids = self._staging[staging_cursor:staging_end]
                staging_cursor = staging_end
                transfers = tuple(
                    BlockTransfer(self._cpu_block(key), gpu_id)
                    for key, gpu_id in zip(keys, gpu_ids, strict=True)
                )
                request_plans.append(
                    RequestLayerBlocks(
                        request_id=request_id,
                        gpu_block_ids=gpu_ids,
                        load=transfers,
                        store=transfers,
                    )
                )
            layer_plans.append(LayerBlockPlan(layer, tuple(request_plans)))

        return BatchBlockPlan(placement.epoch, tuple(layer_plans))

    def _reconcile_resident(self, desired: set[tuple[str, int, int]]) -> None:
        for key in list(self._resident):
            if key not in desired:
                self._free_resident.append(self._resident.pop(key))
        missing = desired - self._resident.keys()
        if len(missing) > len(self._free_resident):
            raise MemoryError(
                f"placement needs {len(missing)} more resident blocks, "
                f"only {len(self._free_resident)} are free"
            )
        for key in sorted(missing):
            self._resident[key] = self._free_resident.pop()

    def _cpu_block(self, key: tuple[str, int, int]) -> int:
        cpu_block = self._cpu_blocks.get(key)
        if cpu_block is None:
            cpu_block = self._next_cpu_block
            self._next_cpu_block += 1
            self._cpu_blocks[key] = cpu_block
        return cpu_block
