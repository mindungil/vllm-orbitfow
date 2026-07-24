# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.orbitflow.config import get_orbitflow_extra_config
from vllm.v1.orbitflow.controller import OrbitFlowController
from vllm.v1.orbitflow.metadata import (
    OrbitFlowConnectorMetadata,
    OrbitFlowWorkerMetadata,
)
from vllm.v1.orbitflow.types import (
    BatchPlacement,
    OrbitFlowConfig,
    RequestPlacement,
    RequestProfile,
)

logger = init_logger(__name__)


class OrbitFlowConnectorScheduler:
    """Scheduler-side placement and profiling state for OrbitFlow."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ):
        extra_config = get_orbitflow_extra_config(vllm_config)
        self._kv_cache_config = kv_cache_config
        self._requests = {}
        self._default_tbt_slo_ms = float(
            extra_config.get("tbt_slo_ms", 100.0)
        )
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
        self._block_size = kv_cache_config.kv_cache_groups[
            0
        ].kv_cache_spec.block_size
        self._num_layers = num_layers
        self._num_staging_banks = kv_cache_config.orbitflow_num_staging_banks
        base, remainder = divmod(
            kv_cache_config.orbitflow_num_staging_blocks,
            self._num_staging_banks,
        )
        self._staging_capacity_by_bank = tuple(
            base + int(bank < remainder)
            for bank in range(self._num_staging_banks)
        )
        self._resident_capacity_blocks = (
            kv_cache_config.num_blocks
            - kv_cache_config.orbitflow_num_staging_blocks
        )

        self._step = 0
        self._request_block_ids: dict[str, list[list[int]]] = {}
        self._prepared_placement: BatchPlacement | None = None
        self._locked_gpu_layers: dict[str, tuple[int, ...]] = {}
        self._pending_migrations: dict[str, tuple[int, ...]] = {}
        self._ready_migrations: dict[
            str, tuple[tuple[int, ...], tuple[int, ...]]
        ] = {}
        self._profile_alpha = float(
            extra_config.get("profile_ewma_alpha", 0.2)
        )
        if not 0 < self._profile_alpha <= 1:
            raise ValueError("profile_ewma_alpha must be in (0, 1]")
        self._measured_layer_ms: float | None = None
        self._measured_bandwidth_bytes_per_ms: float | None = None
        self._last_actual_tbt_ms: float | None = None
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
                max_slo_violations=int(
                    extra_config.get("max_slo_violations", 0)
                ),
                min_replan_interval_steps=int(
                    extra_config.get("min_replan_interval_steps", 1)
                ),
                solver_backend=str(
                    extra_config.get("solver_backend", "auto")
                ),
                solver_timeout_ms=int(
                    extra_config.get("solver_timeout_ms", 20)
                ),
                max_plan_steps=int(extra_config.get("max_plan_steps", 1000)),
            )
        )

    def build_connector_meta(
        self, scheduler_output
    ) -> OrbitFlowConnectorMetadata:
        self._update_request_block_ids(scheduler_output)
        for req_id in scheduler_output.finished_req_ids:
            self._locked_gpu_layers.pop(req_id, None)
            self._pending_migrations.pop(req_id, None)
            self._ready_migrations.pop(req_id, None)
        placement = self._prepared_placement
        self._prepared_placement = None
        if placement is None:
            profiles = self._build_profiles(scheduler_output)
            placement = self._controller.update(profiles, step=self._step)
        placement = self._physical_placement(placement)
        self._step += 1

        active_ids = set(scheduler_output.num_scheduled_tokens)
        active_placement = BatchPlacement(
            epoch=placement.epoch,
            created_at_step=placement.created_at_step,
            expires_at_step=placement.expires_at_step,
            reason=placement.reason,
            requests=tuple(
                request
                for request in placement.requests
                if request.request_id in active_ids
            ),
            paused_request_ids=placement.paused_request_ids,
        )
        return OrbitFlowConnectorMetadata(
            placement=active_placement,
            request_block_ids={
                req_id: tuple(tuple(group) for group in groups)
                for req_id, groups in self._request_block_ids.items()
                if req_id in active_ids
            },
            finished_request_ids=set(scheduler_output.finished_req_ids),
            migration_deposit_layers=self._migration_deposit_layers(),
        )

    def prepare_requests(
        self, requests: list, step: int
    ) -> BatchPlacement:
        profiles = []
        bytes_per_ms = self._measured_bandwidth_bytes_per_ms or (
            self._transfer_bandwidth * 1_000_000
        )
        layer_ms = self._measured_layer_ms or self._compute_ms
        for request in requests:
            self._requests[request.request_id] = request
            num_blocks = max(
                1,
                (request.num_tokens + self._block_size - 1)
                // self._block_size,
            )
            kv_bytes_per_layer = num_blocks * self._page_size_bytes
            params = request.kv_transfer_params or {}
            profiles.append(
                RequestProfile(
                    request_id=request.request_id,
                    kv_bytes_per_layer=kv_bytes_per_layer,
                    compute_ms=layer_ms * self._num_layers,
                    transfer_ms_per_layer=kv_bytes_per_layer / bytes_per_ms,
                    tbt_slo_ms=float(
                        params.get(
                            "orbitflow_tbt_slo_ms",
                            self._default_tbt_slo_ms,
                        )
                    ),
                    deposit_ms=float(
                        params.get("orbitflow_deposit_ms", 0.0)
                    ),
                    kv_growth_bytes_per_step=(
                        self._page_size_bytes / self._block_size
                    ),
                )
            )
        placement = self._controller.update(
            profiles,
            step=step,
            actual_tbt_ms=self._last_actual_tbt_ms,
        )
        physical = self._enforce_physical_capacity(
            self._physical_placement(placement)
        )
        locked_requests = []
        for request in physical.requests:
            req_id = request.request_id
            current = self._locked_gpu_layers.setdefault(
                req_id, request.gpu_layers
            )
            pending = self._pending_migrations.get(req_id)
            if pending is None and current != request.gpu_layers:
                target = request.gpu_layers
                self._pending_migrations[req_id] = target
                logger.info(
                    "OrbitFlow scheduling residence migration for %s: %s -> %s",
                    req_id,
                    current,
                    target,
                )
                if not set(current) - set(target):
                    self._mark_migration_ready(req_id)
                    # Promotion-only migrations need no deposit barrier and
                    # are applied by the core scheduler in this same step.
                    # Keep the worker placement in lockstep with the rewritten
                    # block tables so promoted layers are force-loaded into
                    # their new permanent pages before attention consumes them.
                    current = target
            locked_requests.append(
                RequestPlacement(
                    request_id=request.request_id,
                    gpu_layers=current,
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

    def update_connector_output(self, connector_output) -> None:
        metadata = connector_output.kv_connector_worker_meta
        if isinstance(metadata, OrbitFlowWorkerMetadata):
            self._update_profile(metadata)
            for req_id in metadata.completed_deposit_request_ids:
                self._mark_migration_ready(req_id)

    def _mark_migration_ready(self, request_id: str) -> None:
        target = self._pending_migrations.pop(request_id, None)
        if target is None:
            return
        current = self._locked_gpu_layers.get(request_id, ())
        self._ready_migrations[request_id] = (current, target)
        self._locked_gpu_layers[request_id] = target
        logger.info(
            "OrbitFlow deposit complete for %s: %s -> %s",
            request_id,
            current,
            target,
        )

    def _update_profile(self, metadata: OrbitFlowWorkerMetadata) -> None:
        alpha = self._profile_alpha
        if metadata.compute_layers and metadata.compute_ms > 0:
            sample = metadata.compute_ms / metadata.compute_layers
            self._measured_layer_ms = (
                sample
                if self._measured_layer_ms is None
                else alpha * sample + (1 - alpha) * self._measured_layer_ms
            )
        transfer_ms = metadata.h2d_ms + metadata.d2h_ms
        transfer_bytes = metadata.h2d_bytes + metadata.d2h_bytes
        if transfer_ms > 0 and transfer_bytes > 0:
            sample = transfer_bytes / transfer_ms
            self._measured_bandwidth_bytes_per_ms = (
                sample
                if self._measured_bandwidth_bytes_per_ms is None
                else alpha * sample
                + (1 - alpha) * self._measured_bandwidth_bytes_per_ms
            )
        if self._measured_layer_ms is not None:
            self._last_actual_tbt_ms = (
                self._measured_layer_ms * self._num_layers + transfer_ms
            )

    def take_ready_migrations(
        self,
    ) -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
        ready = self._ready_migrations
        self._ready_migrations = {}
        return ready

    def _migration_deposit_layers(self) -> dict[str, tuple[int, ...]]:
        deposits = {}
        for req_id, target in self._pending_migrations.items():
            current = set(self._locked_gpu_layers.get(req_id, ()))
            demoted = tuple(sorted(current - set(target)))
            if demoted:
                deposits[req_id] = demoted
        return deposits

    def _physical_placement(
        self, placement: BatchPlacement
    ) -> BatchPlacement:
        requests = []
        for request in placement.requests:
            tracked = self._requests.get(request.request_id)
            params = (
                tracked.kv_transfer_params if tracked is not None else None
            ) or {}
            configured = params.get("orbitflow_resident_layers")
            gpu_layers = (
                tuple(int(layer) for layer in configured)
                if configured is not None
                else request.gpu_layers
            )
            if len(set(gpu_layers)) != len(gpu_layers) or any(
                layer < 0 or layer >= self._num_layers
                for layer in gpu_layers
            ):
                raise ValueError(
                    "orbitflow_resident_layers must contain unique valid "
                    "layer indices"
                )
            bytes_per_layer = request.gpu_bytes // max(
                request.num_gpu_layers, 1
            )
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

    def _enforce_physical_capacity(
        self, placement: BatchPlacement
    ) -> BatchPlacement:
        resident_used = 0
        staging_used = [0] * self._num_staging_banks
        paused = list(placement.paused_request_ids)
        already_paused = set(paused)
        for request in placement.requests:
            tracked = self._requests.get(request.request_id)
            if tracked is None:
                continue
            num_blocks = max(
                1,
                (tracked.num_tokens + self._block_size - 1)
                // self._block_size,
            )
            resident_need = num_blocks * request.num_gpu_layers
            offloaded_banks = {
                layer % self._num_staging_banks
                for layer in range(self._num_layers)
                if layer not in request.gpu_layers
            }
            impossible = resident_need > self._resident_capacity_blocks or any(
                num_blocks > self._staging_capacity_by_bank[bank]
                for bank in offloaded_banks
            )
            if impossible:
                raise MemoryError(
                    f"request {request.request_id} cannot fit OrbitFlow's "
                    "resident/staging partition; increase GPU memory or "
                    "num_staging_blocks"
                )
            fits = (
                resident_used + resident_need
                <= self._resident_capacity_blocks
                and all(
                    staging_used[bank] + num_blocks
                    <= self._staging_capacity_by_bank[bank]
                    for bank in offloaded_banks
                )
            )
            if request.request_id in already_paused or not fits:
                if request.request_id not in already_paused:
                    paused.append(request.request_id)
                    already_paused.add(request.request_id)
                continue
            resident_used += resident_need
            for bank in offloaded_banks:
                staging_used[bank] += num_blocks
        return BatchPlacement(
            epoch=placement.epoch,
            created_at_step=placement.created_at_step,
            expires_at_step=placement.expires_at_step,
            reason=placement.reason,
            requests=placement.requests,
            paused_request_ids=tuple(paused),
        )

    def _update_request_block_ids(self, scheduler_output) -> None:
        for request in scheduler_output.scheduled_new_reqs:
            self._request_block_ids[request.req_id] = [
                list(group) for group in request.block_ids
            ]
        cached = scheduler_output.scheduled_cached_reqs
        for req_id, new_groups in zip(
            cached.req_ids, cached.new_block_ids
        ):
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
        profiles = []
        bytes_per_ms = self._measured_bandwidth_bytes_per_ms or (
            self._transfer_bandwidth * 1_000_000
        )
        layer_ms = self._measured_layer_ms or self._compute_ms
        for req_id in set(scheduler_output.num_scheduled_tokens):
            request = self._requests.get(req_id)
            if request is None:
                continue
            num_blocks = max(
                (
                    len(group)
                    for group in self._request_block_ids.get(req_id, ())
                ),
                default=0,
            )
            kv_bytes_per_layer = num_blocks * self._page_size_bytes
            params = request.kv_transfer_params or {}
            profiles.append(
                RequestProfile(
                    request_id=req_id,
                    kv_bytes_per_layer=kv_bytes_per_layer,
                    compute_ms=layer_ms * self._num_layers,
                    transfer_ms_per_layer=(
                        kv_bytes_per_layer / bytes_per_ms
                    ),
                    tbt_slo_ms=float(
                        params.get(
                            "orbitflow_tbt_slo_ms",
                            self._default_tbt_slo_ms,
                        )
                    ),
                    deposit_ms=float(
                        params.get("orbitflow_deposit_ms", 0.0)
                    ),
                    kv_growth_bytes_per_step=(
                        self._page_size_bytes / self._block_size
                    ),
                )
            )
        return profiles

    def on_new_request(self, request) -> None:
        self._requests[request.request_id] = request

    def request_finished(self, request_id: str) -> None:
        self._requests.pop(request_id, None)
        self._request_block_ids.pop(request_id, None)
        self._locked_gpu_layers.pop(request_id, None)
        self._pending_migrations.pop(request_id, None)
        self._ready_migrations.pop(request_id, None)
