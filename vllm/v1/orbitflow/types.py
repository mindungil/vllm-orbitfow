# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from enum import Enum, auto


class PlacementReason(Enum):
    INITIAL = auto()
    EXPIRED = auto()
    BATCH_CHANGED = auto()
    PROFILE_MISMATCH = auto()
    CAPACITY_CHANGED = auto()


@dataclass(frozen=True, slots=True)
class OrbitFlowConfig:
    num_layers: int
    gpu_capacity_bytes: int
    profile_mismatch_ratio: float = 0.2
    max_slo_violations: int = 0
    min_replan_interval_steps: int = 1

    def __post_init__(self) -> None:
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.gpu_capacity_bytes < 0:
            raise ValueError("gpu_capacity_bytes must be non-negative")
        if self.profile_mismatch_ratio < 0:
            raise ValueError("profile_mismatch_ratio must be non-negative")
        if self.max_slo_violations < 0:
            raise ValueError("max_slo_violations must be non-negative")
        if self.min_replan_interval_steps <= 0:
            raise ValueError("min_replan_interval_steps must be positive")


@dataclass(frozen=True, slots=True)
class RequestProfile:
    request_id: str
    kv_bytes_per_layer: int
    compute_ms: float
    transfer_ms_per_layer: float
    tbt_slo_ms: float
    deposit_ms: float = 0.0

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.kv_bytes_per_layer < 0:
            raise ValueError("kv_bytes_per_layer must be non-negative")
        if self.compute_ms < 0 or self.transfer_ms_per_layer < 0:
            raise ValueError("latencies must be non-negative")
        if self.tbt_slo_ms <= 0:
            raise ValueError("tbt_slo_ms must be positive")
        if self.deposit_ms < 0:
            raise ValueError("deposit_ms must be non-negative")


@dataclass(frozen=True, slots=True)
class RequestPlacement:
    request_id: str
    gpu_layers: tuple[int, ...]
    predicted_tbt_ms: float
    gpu_bytes: int
    violates_slo: bool

    @property
    def num_gpu_layers(self) -> int:
        return len(self.gpu_layers)


@dataclass(frozen=True, slots=True)
class BatchPlacement:
    epoch: int
    created_at_step: int
    expires_at_step: int
    reason: PlacementReason
    requests: tuple[RequestPlacement, ...]
    paused_request_ids: tuple[str, ...] = ()

    @property
    def gpu_bytes(self) -> int:
        return sum(request.gpu_bytes for request in self.requests)

    def for_request(self, request_id: str) -> RequestPlacement:
        for request in self.requests:
            if request.request_id == request_id:
                return request
        raise KeyError(request_id)
