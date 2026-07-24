# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import OffloadingSpec
from vllm.v1.orbitflow.config import get_orbitflow_extra_config
from vllm.v1.orbitflow.controller import OrbitFlowController
from vllm.v1.orbitflow.metadata import OrbitFlowConnectorMetadata
from vllm.v1.orbitflow.types import (
    BatchPlacement,
    OrbitFlowConfig,
    RequestPlacement,
    RequestProfile,
)

logger = init_logger(__name__)


class OrbitFlowConnectorScheduler(OffloadingConnectorScheduler):
    """Adds adaptive request-wise layer placement to V1 offloading metadata."""

    def __init__(
        self,
        spec: OffloadingSpec,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ):
        super().__init__(spec, vllm_config, kv_cache_config)
        extra_config = get_orbitflow_extra_config(vllm_config)
        self._default_tbt_slo_ms = float(extra_config.get("tbt_slo_ms", 100.0))
        self._compute_ms = float(extra_config.get("layer_compute_ms", 0.1))
        self._transfer_bandwidth = float(
            extra_config.get("transfer_bandwidth_gbps", 12.0)
        )
        if self._transfer_bandwidth <= 0:
            raise ValueError("transfer_bandwidth_gbps must be positive")

        num_layers = len(kv_cache_config.kv_cache_groups)
        page_sizes = {
            group.kv_cache_spec.page_size_bytes
            for group in kv_cache_config.kv_cache_groups
        }
        if len(page_sizes) != 1:
            raise ValueError("OrbitFlow requires a uniform KV page size")
        self._page_size_bytes = page_sizes.pop()
        self._step = 0
        self._num_resident_layers = int(
            extra_config.get("num_resident_layers", num_layers)
        )
        self._request_block_ids: dict[str, list[list[int]]] = {}
        self._prepared_placement: BatchPlacement | None = None
        self._locked_gpu_layers: dict[str, tuple[int, ...]] = {}
        self._pending_migrations: dict[str, tuple[tuple[int, ...], int]] = {}
        self._ready_migrations: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
        gpu_capacity_bytes = int(
            extra_config.get(
                "gpu_capacity_bytes",
                kv_cache_config.num_blocks * self._page_size_bytes,
            )
        )
        if gpu_capacity_bytes < 0:
            raise ValueError("gpu_capacity_bytes must be non-negative")
        self._controller = OrbitFlowController(
            OrbitFlowConfig(
                num_layers=num_layers,
                gpu_capacity_bytes=gpu_capacity_bytes,
                profile_mismatch_ratio=float(
                    extra_config.get("profile_mismatch_ratio", 0.2)
                ),
                max_slo_violations=int(extra_config.get("max_slo_violations", 0)),
                min_replan_interval_steps=int(
                    extra_config.get("min_replan_interval_steps", 1)
                ),
            )
        )

    def build_connector_meta(self, scheduler_output) -> OrbitFlowConnectorMetadata:
        offload_meta = super().build_connector_meta(scheduler_output)
        self._update_request_block_ids(scheduler_output)
        for req_id in scheduler_output.finished_req_ids:
            self._locked_gpu_layers.pop(req_id, None)
            self._pending_migrations.pop(req_id, None)
        placement = self._prepared_placement
        self._prepared_placement = None
        if placement is None:
            profiles = self._build_profiles(scheduler_output)
            placement = self._controller.update(profiles, step=self._step)
        placement = self._physical_placement(placement)
        self._step += 1
        return OrbitFlowConnectorMetadata(
            load_jobs=offload_meta.load_jobs,
            store_jobs=offload_meta.store_jobs,
            jobs_to_flush=offload_meta.jobs_to_flush,
            placement=placement,
            request_block_ids={
                req_id: tuple(tuple(group) for group in groups)
                for req_id, groups in self._request_block_ids.items()
                if req_id in scheduler_output.num_scheduled_tokens
            },
            finished_request_ids=set(scheduler_output.finished_req_ids),
            migration_deposit_layers=self._migration_deposit_layers(),
        )

    def prepare_requests(self, requests: list, step: int) -> BatchPlacement:
        profiles = []
        bytes_per_ms = self._transfer_bandwidth * 1_000_000
        block_size = self.config.kv_group_configs[0].tokens_per_block
        for request in requests:
            num_blocks = max(1, (request.num_tokens + block_size - 1) // block_size)
            kv_bytes_per_layer = num_blocks * self._page_size_bytes
            params = request.kv_transfer_params or {}
            profiles.append(
                RequestProfile(
                    request_id=request.request_id,
                    kv_bytes_per_layer=kv_bytes_per_layer,
                    compute_ms=self._compute_ms * self._controller.config.num_layers,
                    transfer_ms_per_layer=kv_bytes_per_layer / bytes_per_ms,
                    tbt_slo_ms=float(
                        params.get("orbitflow_tbt_slo_ms", self._default_tbt_slo_ms)
                    ),
                    deposit_ms=float(params.get("orbitflow_deposit_ms", 0.0)),
                )
            )
        placement = self._controller.update(profiles, step=step)
        physical = self._physical_placement(placement)
        locked_requests = []
        for request in physical.requests:
            req_id = request.request_id
            current = self._locked_gpu_layers.setdefault(req_id, request.gpu_layers)
            pending = self._pending_migrations.get(req_id)
            if pending is not None and step >= pending[1]:
                target = pending[0]
                self._ready_migrations[req_id] = (current, target)
                self._locked_gpu_layers[req_id] = target
                del self._pending_migrations[req_id]
                logger.info(
                    "OrbitFlow applying residence migration for %s: %s -> %s",
                    req_id,
                    current,
                    target,
                )
                current = target
            elif pending is None and current != request.gpu_layers:
                self._pending_migrations[req_id] = (request.gpu_layers, step + 1)
                logger.info(
                    "OrbitFlow scheduling residence migration for %s: %s -> %s",
                    req_id,
                    current,
                    request.gpu_layers,
                )
            gpu_layers = current
            locked_requests.append(
                RequestPlacement(
                    request_id=request.request_id,
                    gpu_layers=gpu_layers,
                    predicted_tbt_ms=request.predicted_tbt_ms,
                    gpu_bytes=request.gpu_bytes,
                    violates_slo=request.violates_slo,
                )
            )
        self._prepared_placement = BatchPlacement(
            epoch=physical.epoch,
            created_at_step=physical.created_at_step,
            expires_at_step=physical.expires_at_step,
            reason=physical.reason,
            requests=tuple(locked_requests),
            paused_request_ids=physical.paused_request_ids,
        )
        if physical.paused_request_ids:
            logger.info(
                "OrbitFlow paused requests at step %d: %s",
                step,
                physical.paused_request_ids,
            )
        return self._prepared_placement

    def take_ready_migrations(
        self,
    ) -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
        ready = self._ready_migrations
        self._ready_migrations = {}
        return ready

    def _migration_deposit_layers(self) -> dict[str, tuple[int, ...]]:
        deposits = {}
        for req_id, (target, _) in self._pending_migrations.items():
            current = set(self._locked_gpu_layers.get(req_id, ()))
            demoted = tuple(sorted(current - set(target)))
            if demoted:
                deposits[req_id] = demoted
        return deposits

    def _physical_placement(self, placement: BatchPlacement) -> BatchPlacement:
        requests = []
        for request in placement.requests:
            req_status = self._req_status.get(request.request_id)
            params = (
                req_status.req.kv_transfer_params if req_status is not None else None
            ) or {}
            configured = params.get("orbitflow_resident_layers")
            gpu_layers = (
                tuple(int(layer) for layer in configured)
                if configured is not None
                else request.gpu_layers
            )
            bytes_per_layer = request.gpu_bytes // max(request.num_gpu_layers, 1)
            requests.append(
                RequestPlacement(
                    request_id=request.request_id,
                    gpu_layers=gpu_layers,
                    predicted_tbt_ms=request.predicted_tbt_ms,
                    gpu_bytes=bytes_per_layer * len(gpu_layers),
                    violates_slo=request.violates_slo,
                )
            )
        return BatchPlacement(
            epoch=placement.epoch,
            created_at_step=placement.created_at_step,
            expires_at_step=placement.expires_at_step,
            reason=placement.reason,
            requests=tuple(requests),
            paused_request_ids=placement.paused_request_ids,
        )

    def _update_request_block_ids(self, scheduler_output) -> None:
        for request in scheduler_output.scheduled_new_reqs:
            self._request_block_ids[request.req_id] = [
                list(group) for group in request.block_ids
            ]
        cached = scheduler_output.scheduled_cached_reqs
        for req_id, new_groups in zip(cached.req_ids, cached.new_block_ids):
            if new_groups is None:
                continue
            groups = self._request_block_ids.setdefault(
                req_id, [[] for _ in new_groups]
            )
            if req_id in cached.resumed_req_ids:
                groups[:] = [list(group) for group in new_groups]
            else:
                for current, new in zip(groups, new_groups):
                    current.extend(new)
        for req_id in scheduler_output.finished_req_ids:
            self._request_block_ids.pop(req_id, None)

    def _build_profiles(self, scheduler_output) -> list[RequestProfile]:
        active_req_ids = set(scheduler_output.num_scheduled_tokens)
        profiles = []
        bytes_per_ms = self._transfer_bandwidth * 1_000_000
        for req_id in active_req_ids:
            req_status = self._req_status.get(req_id)
            if req_status is None:
                continue
            request = req_status.req
            num_blocks = max(
                (len(group_state.block_ids) for group_state in req_status.group_states),
                default=0,
            )
            kv_bytes_per_layer = num_blocks * self._page_size_bytes
            transfer_ms = kv_bytes_per_layer / bytes_per_ms
            params = request.kv_transfer_params or {}
            profiles.append(
                RequestProfile(
                    request_id=req_id,
                    kv_bytes_per_layer=kv_bytes_per_layer,
                    compute_ms=self._compute_ms * self._controller.config.num_layers,
                    transfer_ms_per_layer=transfer_ms,
                    tbt_slo_ms=float(
                        params.get(
                            "orbitflow_tbt_slo_ms",
                            self._default_tbt_slo_ms,
                        )
                    ),
                    deposit_ms=float(params.get("orbitflow_deposit_ms", 0.0)),
                )
            )
        return profiles
