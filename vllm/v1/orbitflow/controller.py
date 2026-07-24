# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.orbitflow.planner import OrbitFlowPlanner
from vllm.v1.orbitflow.types import (
    BatchPlacement,
    OrbitFlowConfig,
    PlacementReason,
    RequestProfile,
)


class OrbitFlowController:
    """Owns placement epochs and OrbitFlow replanning triggers."""

    def __init__(self, config: OrbitFlowConfig):
        self.config = config
        self.planner = OrbitFlowPlanner(config)
        self.placement: BatchPlacement | None = None
        self._epoch = 0
        self._request_ids: frozenset[str] = frozenset()

    def update(
        self,
        profiles: list[RequestProfile],
        *,
        step: int,
        actual_tbt_ms: float | None = None,
        available_gpu_bytes: int | None = None,
    ) -> BatchPlacement:
        reason = self._replan_reason(
            profiles,
            step=step,
            actual_tbt_ms=actual_tbt_ms,
            available_gpu_bytes=available_gpu_bytes,
        )
        if reason is None:
            assert self.placement is not None
            return self.placement

        planner = self.planner
        if (
            available_gpu_bytes is not None
            and available_gpu_bytes != self.config.gpu_capacity_bytes
        ):
            planner = OrbitFlowPlanner(
                OrbitFlowConfig(
                    num_layers=self.config.num_layers,
                    gpu_capacity_bytes=available_gpu_bytes,
                    profile_mismatch_ratio=self.config.profile_mismatch_ratio,
                    max_slo_violations=self.config.max_slo_violations,
                    min_replan_interval_steps=self.config.min_replan_interval_steps,
                )
            )

        self._epoch += 1
        self.placement = planner.plan(
            profiles, epoch=self._epoch, step=step, reason=reason
        )
        self._request_ids = frozenset(profile.request_id for profile in profiles)
        return self.placement

    def _replan_reason(
        self,
        profiles: list[RequestProfile],
        *,
        step: int,
        actual_tbt_ms: float | None,
        available_gpu_bytes: int | None,
    ) -> PlacementReason | None:
        if self.placement is None:
            return PlacementReason.INITIAL
        if (
            step - self.placement.created_at_step
            < self.config.min_replan_interval_steps
        ):
            return None

        request_ids = frozenset(profile.request_id for profile in profiles)
        if request_ids != self._request_ids:
            return PlacementReason.BATCH_CHANGED
        if available_gpu_bytes is not None and (
            available_gpu_bytes != self.config.gpu_capacity_bytes
        ):
            return PlacementReason.CAPACITY_CHANGED
        if actual_tbt_ms is not None:
            predicted = max(
                (request.predicted_tbt_ms for request in self.placement.requests),
                default=0.0,
            )
            mismatch = abs(actual_tbt_ms - predicted) / max(predicted, 1e-6)
            if mismatch > self.config.profile_mismatch_ratio:
                return PlacementReason.PROFILE_MISMATCH
        if step >= self.placement.expires_at_step:
            return PlacementReason.EXPIRED
        return None
