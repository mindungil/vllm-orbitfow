# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlock


@dataclass(frozen=True, slots=True)
class StagingAssignment:
    request_id: str
    bank: int
    block_ids: tuple[int, ...]


class OrbitFlowStagingBlockManager:
    """Owns physical GPU blocks reused by every non-resident layer.

    Assignments are request-scoped so requests within one attention layer do
    not alias each other. Different layers deliberately receive the same
    assignment and are serialized by the attention-layer transfer hooks.
    """

    def __init__(
        self,
        block_pool: BlockPool,
        num_staging_blocks: int,
        num_banks: int = 1,
    ) -> None:
        if num_staging_blocks <= 0:
            raise ValueError("num_staging_blocks must be positive")
        if num_staging_blocks >= block_pool.num_gpu_blocks:
            raise ValueError("staging must leave at least one resident block")
        if num_banks <= 0 or num_banks > num_staging_blocks:
            raise ValueError("num_banks must be between 1 and num_staging_blocks")
        self._block_pool = block_pool
        self.num_banks = num_banks
        self._blocks = tuple(block_pool.get_new_blocks(num_staging_blocks))
        self._free_ids: list[list[int]] = [[] for _ in range(num_banks)]
        block_ids = sorted(block.block_id for block in self._blocks)
        base, remainder = divmod(len(block_ids), num_banks)
        offset = 0
        for bank in range(num_banks):
            size = base + int(bank < remainder)
            # pop() then returns ascending IDs, preserving contiguous runs.
            self._free_ids[bank] = list(
                reversed(block_ids[offset : offset + size])
            )
            offset += size
        self._assignments: dict[tuple[str, int], tuple[int, ...]] = {}

    @property
    def block_ids(self) -> tuple[int, ...]:
        return tuple(block.block_id for block in self._blocks)

    @property
    def num_free_blocks(self) -> int:
        return min(map(len, self._free_ids))

    def assign(
        self, request_id: str, num_blocks: int, bank: int = 0
    ) -> StagingAssignment:
        if num_blocks < 0:
            raise ValueError("num_blocks must be non-negative")
        if not 0 <= bank < self.num_banks:
            raise ValueError(f"invalid staging bank {bank}")
        key = (request_id, bank)
        current = self._assignments.get(key, ())
        missing = num_blocks - len(current)
        free_ids = self._free_ids[bank]
        if missing > len(free_ids):
            raise MemoryError(
                f"request {request_id} needs {missing} more staging blocks "
                f"in bank {bank}; only {len(free_ids)} are free"
            )
        if missing > 0:
            current += tuple(free_ids.pop() for _ in range(missing))
            self._assignments[key] = current
        return StagingAssignment(request_id, bank, current[:num_blocks])

    def release(self, request_id: str) -> None:
        for bank in range(self.num_banks):
            self.release_bank(request_id, bank)

    def release_bank(self, request_id: str, bank: int) -> None:
        if not 0 <= bank < self.num_banks:
            raise ValueError(f"invalid staging bank {bank}")
        block_ids = self._assignments.pop((request_id, bank), ())
        self._free_ids[bank].extend(reversed(block_ids))

    def as_virtual_blocks(
        self, request_id: str, num_blocks: int, bank: int = 0
    ) -> list[KVCacheBlock]:
        assignment = self.assign(request_id, num_blocks, bank)
        # These metadata objects are intentionally detached from BlockPool.
        # The pool owns one reserved object per physical staging ID; virtual
        # objects only carry IDs into per-layer block tables.
        return [KVCacheBlock(block_id) for block_id in assignment.block_ids]
