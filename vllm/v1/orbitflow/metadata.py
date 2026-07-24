# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
)
from vllm.v1.orbitflow.types import BatchPlacement


@dataclass
class OrbitFlowConnectorMetadata(OffloadingConnectorMetadata):
    placement: BatchPlacement | None = None
    request_block_ids: dict[str, tuple[tuple[int, ...], ...]] | None = None
    finished_request_ids: set[str] | None = None
    migration_deposit_layers: dict[str, tuple[int, ...]] | None = None
