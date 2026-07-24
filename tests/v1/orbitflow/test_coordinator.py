# SPDX-License-Identifier: Apache-2.0

import torch

from vllm.v1.core.kv_cache_coordinator import (
    OrbitFlowKVCacheCoordinator,
    get_kv_cache_coordinator,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)


def _config() -> KVCacheConfig:
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.float16,
    )
    return KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([f"layer.{i}"], spec) for i in range(5)],
        orbitflow_num_staging_blocks=9,
        orbitflow_num_resident_layers=1,
        orbitflow_num_staging_banks=3,
    )


def _coordinator() -> OrbitFlowKVCacheCoordinator:
    coordinator = get_kv_cache_coordinator(
        kv_cache_config=_config(),
        max_model_len=256,
        max_in_flight_tokens=256,
        use_eagle=False,
        enable_caching=False,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        scheduler_block_size=16,
        hash_block_size=16,
    )
    assert isinstance(coordinator, OrbitFlowKVCacheCoordinator)
    return coordinator


def test_offloaded_groups_reuse_staging_ids():
    coordinator = _coordinator()
    blocks = coordinator.allocate_new_blocks("r", 32, 32)

    assert len(blocks[0]) == 2
    assert [b.block_id for b in blocks[1]] != [b.block_id for b in blocks[2]]
    assert [b.block_id for b in blocks[1]] == [b.block_id for b in blocks[4]]
    assert {b.block_id for b in blocks[0]}.isdisjoint({b.block_id for b in blocks[1]})


def test_staging_is_released_with_request():
    coordinator = _coordinator()
    first = coordinator.allocate_new_blocks("a", 32, 32)
    first_ids = [b.block_id for b in first[1]]
    coordinator.free("a")
    second = coordinator.allocate_new_blocks("b", 32, 32)

    assert first_ids == [b.block_id for b in second[1]]


def test_request_specific_residence():
    coordinator = _coordinator()
    coordinator.set_request_resident_layers("a", frozenset({0, 2}))
    coordinator.set_request_resident_layers("b", frozenset({1, 3}))

    a = coordinator.allocate_new_blocks("a", 16, 16)
    b = coordinator.allocate_new_blocks("b", 16, 16)

    staging_ids = set(coordinator.staging.block_ids)
    assert a[0][0].block_id not in staging_ids
    assert a[1][0].block_id in staging_ids
    assert a[2][0].block_id not in staging_ids
    assert b[0][0].block_id in staging_ids
    assert b[1][0].block_id not in staging_ids


def test_residence_migration_rewrites_physical_ownership():
    coordinator = _coordinator()
    coordinator.set_request_resident_layers("r", frozenset({0, 2}))
    before = coordinator.allocate_new_blocks("r", 16, 16)
    staging_ids = set(coordinator.staging.block_ids)
    old_layer0 = before[0][0].block_id

    coordinator.migrate_request("r", frozenset({1, 2}))
    after = coordinator.get_blocks("r")

    assert after[0][0].block_id in staging_ids
    assert after[1][0].block_id not in staging_ids
    assert after[2][0].block_id == before[2][0].block_id
    assert old_layer0 != after[0][0].block_id


def test_migration_releases_unused_staging_bank():
    coordinator = _coordinator()
    coordinator.set_request_resident_layers("r", frozenset({0, 2, 3, 4}))
    coordinator.allocate_new_blocks("r", 16, 16)
    free_before = coordinator.staging.num_free_blocks

    coordinator.migrate_request("r", frozenset(range(5)))

    assert coordinator.staging.num_free_blocks == free_before + 1
