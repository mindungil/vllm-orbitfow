# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    build_offloading_config,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.factory import OffloadingSpecFactory
from vllm.v1.orbitflow.config import get_orbitflow_extra_config
from vllm.v1.orbitflow.layer_store import OrbitFlowLayerStore
from vllm.v1.orbitflow.metadata import OrbitFlowConnectorMetadata
from vllm.v1.orbitflow.runtime import (
    LayerTransferAction,
    OrbitFlowLayerPipeline,
)
from vllm.v1.orbitflow.scheduler import OrbitFlowConnectorScheduler


class OrbitFlowConnector(OffloadingConnector):
    """V1 OrbitFlow connector using the native offloading backend."""

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
        self._layer_pipeline: OrbitFlowLayerPipeline | None = None
        self._layer_store: OrbitFlowLayerStore | None = None
        if role == KVConnectorRole.WORKER:
            layer_names = tuple(
                group.layer_names[0] for group in kv_cache_config.kv_cache_groups
            )
            extra_config = get_orbitflow_extra_config(vllm_config)
            prefetch_distance = int(extra_config.get("prefetch_distance", 2))
            num_resident_layers = int(
                extra_config.get("num_resident_layers", len(layer_names))
            )
            self._layer_store = OrbitFlowLayerStore(
                layer_names,
                num_resident_layers,
                kv_cache_config.num_blocks,
                bool(extra_config.get("validate_transfers", False)),
            )
            self._layer_pipeline = OrbitFlowLayerPipeline(
                layer_names,
                prefetch_distance=prefetch_distance,
                load=self._start_layer_load,
                wait=self._wait_layer_load,
                store=self._store_layer,
            )
        if role == KVConnectorRole.SCHEDULER:
            assert self.connector_scheduler is not None
            self.connector_scheduler.shutdown()
            offloading_config = build_offloading_config(vllm_config, kv_cache_config)
            spec = OffloadingSpecFactory.create_spec(offloading_config)
            self.connector_scheduler = OrbitFlowConnectorScheduler(
                spec, vllm_config, kv_cache_config
            )

    def start_load_kv(self, forward_context, **kwargs) -> None:
        super().start_load_kv(forward_context, **kwargs)
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

    def register_kv_caches(self, kv_caches) -> None:
        super().register_kv_caches(kv_caches)
        if self._layer_store is not None:
            self._layer_store.register_kv_caches(kv_caches)

    def wait_for_layer_load(self, layer_name: str) -> None:
        if self._layer_pipeline is not None:
            self._layer_pipeline.wait_for_layer(layer_name)
        if self._layer_store is not None:
            self._layer_store.start_load_layer(layer_name)
            self._layer_store.wait_for_layer(layer_name)

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs) -> None:
        if self._layer_store is not None:
            self._layer_store.store_layer(layer_name)

    def wait_for_save(self):
        if self._layer_store is not None:
            self._layer_store.wait_for_pending_deposits()

    def _start_layer_load(self, action: LayerTransferAction) -> None:
        if self._layer_store is not None:
            self._layer_store.start_load_layer(
                self._layer_pipeline.layer_names[action.layer_index]
            )

    def _wait_layer_load(self, action: LayerTransferAction) -> None:
        assert self.connector_worker is not None
        self.connector_worker.wait_for_loads(set(action.request_ids))
        if self._layer_store is not None:
            self._layer_store.wait_for_layer(
                self._layer_pipeline.layer_names[action.layer_index]
            )

    def _store_layer(self, action: LayerTransferAction) -> None:
        # Stores are collected by the native connector after sampling, when KV
        # written by all attention layers is stable.
        if self._layer_store is not None:
            self._layer_store.store_layer(
                self._layer_pipeline.layer_names[action.layer_index]
            )

    def prepare_orbitflow_placement(self, requests, step):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.prepare_requests(list(requests), step)

    def take_ready_orbitflow_migrations(self):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.take_ready_migrations()

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
        if parallel_config.tensor_parallel_size != 1:
            raise NotImplementedError(
                "OrbitFlow V1 currently supports a single GPU only"
            )

    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config) -> bool:
        return True
