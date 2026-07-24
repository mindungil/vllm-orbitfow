# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)
from vllm.v1.orbitflow.types import BatchPlacement


@dataclass
class OrbitFlowConnectorMetadata(KVConnectorMetadata):
    placement: BatchPlacement | None = None
    request_block_ids: dict[str, tuple[tuple[int, ...], ...]] | None = None
    finished_request_ids: set[str] | None = None
    migration_deposit_layers: dict[str, tuple[int, ...]] | None = None


@dataclass
class OrbitFlowWorkerMetadata(KVConnectorWorkerMetadata):
    compute_ms: float = 0.0
    compute_layers: int = 0
    h2d_ms: float = 0.0
    h2d_bytes: int = 0
    d2h_ms: float = 0.0
    d2h_bytes: int = 0
    completed_deposit_request_ids: tuple[str, ...] = ()

    def aggregate(
        self, other: KVConnectorWorkerMetadata
    ) -> "OrbitFlowWorkerMetadata":
        assert isinstance(other, OrbitFlowWorkerMetadata)
        return OrbitFlowWorkerMetadata(
            compute_ms=self.compute_ms + other.compute_ms,
            compute_layers=self.compute_layers + other.compute_layers,
            h2d_ms=max(self.h2d_ms, other.h2d_ms),
            h2d_bytes=max(self.h2d_bytes, other.h2d_bytes),
            d2h_ms=max(self.d2h_ms, other.d2h_ms),
            d2h_bytes=max(self.d2h_bytes, other.d2h_bytes),
            completed_deposit_request_ids=tuple(
                sorted(
                    set(self.completed_deposit_request_ids)
                    & set(other.completed_deposit_request_ids)
                )
            ),
        )
