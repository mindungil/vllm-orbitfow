# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.forward_context import ForwardContext
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.orbitflow.config import get_orbitflow_extra_config
from vllm.v1.orbitflow.layer_store import OrbitFlowLayerStore
from vllm.v1.orbitflow.metadata import (
    OrbitFlowConnectorMetadata,
    OrbitFlowWorkerMetadata,
)
from vllm.v1.orbitflow.runtime import (
    LayerTransferAction,
    OrbitFlowLayerPipeline,
)
from vllm.v1.orbitflow.scheduler import OrbitFlowConnectorScheduler
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request


class OrbitFlowConnector(KVConnectorBase_V1, SupportsHMA):
    """Layer-wise V1 connector for OrbitFlow's reusable staging pages."""

    @property
    def load_before_kv_update(self) -> bool:
        return True

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        self._validate_config(vllm_config)
        super().__init__(vllm_config, role, kv_cache_config)
        self.connector_scheduler: OrbitFlowConnectorScheduler | None = None
        self._layer_pipeline: OrbitFlowLayerPipeline | None = None
        self._layer_store: OrbitFlowLayerStore | None = None

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = OrbitFlowConnectorScheduler(
                vllm_config, kv_cache_config
            )
            return

        layer_names = tuple(
            group.layer_names[0] for group in kv_cache_config.kv_cache_groups
        )
        extra_config = get_orbitflow_extra_config(vllm_config)
        prefetch_distance = int(extra_config.get("prefetch_distance", 2))
        num_resident_layers = int(
            extra_config.get("num_resident_layers", len(layer_names))
        )
        cpu_cache_bytes = extra_config.get("cpu_cache_bytes")
        if cpu_cache_bytes is None:
            cpu_cache_bytes = extra_config.get("cpu_bytes_to_use_per_rank")
        if cpu_cache_bytes is None and "cpu_bytes_to_use" in extra_config:
            cpu_cache_bytes = int(extra_config["cpu_bytes_to_use"]) // (
                vllm_config.parallel_config.world_size
            )
        self._layer_store = OrbitFlowLayerStore(
            layer_names,
            num_resident_layers,
            kv_cache_config.num_blocks,
            bool(extra_config.get("validate_transfers", False)),
            int(cpu_cache_bytes) if cpu_cache_bytes is not None else None,
            extra_config.get("nvme_path"),
            int(extra_config.get("nvme_bytes", 0)),
            bool(extra_config.get("coalesce_transfers", True)),
        )
        self._layer_pipeline = OrbitFlowLayerPipeline(
            layer_names,
            prefetch_distance=prefetch_distance,
            load=self._start_layer_load,
            wait=self._wait_layer_load,
            store=self._store_layer,
        )

    # Worker side.

    def start_load_kv(
        self, forward_context: ForwardContext, **kwargs: Any
    ) -> None:
        assert isinstance(self._connector_metadata, OrbitFlowConnectorMetadata)
        if (
            self._layer_store is not None
            and self._connector_metadata.request_block_ids is not None
        ):
            placement = self._connector_metadata.placement
            self._layer_store.begin_batch(
                self._connector_metadata.request_block_ids,
                self._connector_metadata.finished_request_ids or set(),
                {
                    request.request_id: frozenset(request.gpu_layers)
                    for request in placement.requests
                }
                if placement is not None
                else {},
                self._connector_metadata.migration_deposit_layers or {},
            )
        if (
            self._layer_pipeline is not None
            and self._connector_metadata.placement is not None
        ):
            self._layer_pipeline.begin_batch(self._connector_metadata.placement)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if self._layer_store is not None:
            self._layer_store.register_kv_caches(kv_caches)

    def wait_for_layer_load(self, layer_name: str) -> None:
        if self._layer_pipeline is not None:
            self._layer_pipeline.wait_for_layer(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        if self._layer_store is not None:
            self._layer_store.store_layer(layer_name)

    def wait_for_save(self) -> None:
        if self._layer_store is not None:
            self._layer_store.wait_for_pending_saves()

    def build_connector_worker_meta(self) -> OrbitFlowWorkerMetadata | None:
        if self._layer_store is None:
            return None
        profile = self._layer_store.collect_profile()
        return OrbitFlowWorkerMetadata(
            compute_ms=profile.compute_ms,
            compute_layers=profile.compute_layers,
            h2d_ms=profile.h2d_ms,
            h2d_bytes=profile.h2d_bytes,
            d2h_ms=profile.d2h_ms,
            d2h_bytes=profile.d2h_bytes,
            completed_deposit_request_ids=(
                self._layer_store.take_completed_deposit_request_ids()
            ),
        )

    def _start_layer_load(self, action: LayerTransferAction) -> None:
        if self._layer_store is not None:
            assert self._layer_pipeline is not None
            self._layer_store.start_load_layer(
                self._layer_pipeline.layer_names[action.layer_index]
            )

    def _wait_layer_load(self, action: LayerTransferAction) -> None:
        if self._layer_store is not None:
            assert self._layer_pipeline is not None
            self._layer_store.wait_for_layer(
                self._layer_pipeline.layer_names[action.layer_index]
            )

    def _store_layer(self, action: LayerTransferAction) -> None:
        if self._layer_store is not None:
            assert self._layer_pipeline is not None
            self._layer_store.store_layer(
                self._layer_pipeline.layer_names[action.layer_index]
            )

    # Scheduler side.

    def prepare_orbitflow_placement(self, requests, step):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.prepare_requests(list(requests), step)

    def take_ready_orbitflow_migrations(self):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.take_ready_migrations()

    def on_new_request(self, request: Request) -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.on_new_request(request)

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        return 0, False

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        return None

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def update_connector_output(
        self, connector_output: KVConnectorOutput
    ) -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.update_connector_output(connector_output)

    def request_finished(
        self, request: Request, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        if self.connector_scheduler is not None:
            self.connector_scheduler.request_finished(request.request_id)
        return False, None

    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        return self.request_finished(request, [])

    def take_events(self) -> Iterable[KVCacheEvent]:
        return ()

    @staticmethod
    def _validate_config(vllm_config: VllmConfig) -> None:
        vllm_config.scheduler_config.async_scheduling = False
        parallel_config = vllm_config.parallel_config
        if parallel_config.pipeline_parallel_size != 1:
            raise NotImplementedError(
                "OrbitFlow does not support pipeline parallelism yet"
            )
        if vllm_config.speculative_config is not None:
            raise NotImplementedError(
                "OrbitFlow does not support speculative decoding yet"
            )
        if vllm_config.cache_config.enable_prefix_caching:
            raise NotImplementedError(
                "OrbitFlow requires prefix caching to be disabled"
            )

    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config) -> bool:
        return True
