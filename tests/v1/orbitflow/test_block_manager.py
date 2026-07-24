# SPDX-License-Identifier: Apache-2.0

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.orbitflow.block_manager import OrbitFlowStagingBlockManager


def _pool(num_blocks: int = 16) -> BlockPool:
    return BlockPool(
        num_gpu_blocks=num_blocks,
        enable_caching=False,
        hash_block_size=16,
        enable_kv_cache_events=False,
    )


def test_staging_assignments_are_request_disjoint_and_layer_reusable():
    pool = _pool()
    manager = OrbitFlowStagingBlockManager(pool, 6)

    a = manager.assign("a", 2)
    b = manager.assign("b", 3)
    assert set(a.block_ids).isdisjoint(b.block_ids)
    assert manager.assign("a", 2) == a
    assert pool.get_num_free_blocks() == 9


def test_staging_release_reuses_physical_blocks():
    manager = OrbitFlowStagingBlockManager(_pool(), 4)
    old = manager.assign("a", 3)
    manager.release("a")
    new = manager.assign("b", 3)
    assert set(old.block_ids) == set(new.block_ids)


def test_ring_banks_are_disjoint_and_reused_by_bank():
    manager = OrbitFlowStagingBlockManager(_pool(20), 9, num_banks=3)
    bank0 = manager.assign("a", 2, bank=0)
    bank1 = manager.assign("a", 2, bank=1)
    assert set(bank0.block_ids).isdisjoint(bank1.block_ids)
    assert manager.assign("a", 2, bank=0) == bank0
