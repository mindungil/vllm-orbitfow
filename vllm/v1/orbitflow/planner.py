# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

from vllm.v1.orbitflow.types import (
    BatchPlacement,
    OrbitFlowConfig,
    PlacementReason,
    RequestPlacement,
    RequestProfile,
)


@dataclass(frozen=True, slots=True)
class _Candidate:
    num_gpu_layers: int
    gpu_bytes: int
    predicted_tbt_ms: float
    violates_slo: bool


class OrbitFlowPlanner:
    """Plans request-wise layer residency under a GPU KV capacity bound."""

    def __init__(self, config: OrbitFlowConfig):
        self.config = config

    def plan(
        self,
        profiles: list[RequestProfile],
        *,
        epoch: int,
        step: int,
        reason: PlacementReason,
    ) -> BatchPlacement:
        if len({profile.request_id for profile in profiles}) != len(profiles):
            raise ValueError("request ids must be unique")

        active = list(profiles)
        paused: list[str] = []
        selected = self._select(active)
        while selected is None and active:
            victim = max(
                active,
                key=lambda profile: (
                    profile.kv_bytes_per_layer,
                    profile.transfer_ms_per_layer,
                    profile.request_id,
                ),
            )
            active.remove(victim)
            paused.append(victim.request_id)
            selected = self._select(active)

        placements = tuple(
            self._to_placement(profile, candidate)
            for profile, candidate in zip(active, selected or (), strict=True)
        )
        expires_at = self._estimate_expiry(step, active, placements)
        return BatchPlacement(
            epoch=epoch,
            created_at_step=step,
            expires_at_step=expires_at,
            reason=reason,
            requests=placements,
            paused_request_ids=tuple(paused),
        )

    def _select(self, profiles: list[RequestProfile]) -> tuple[_Candidate, ...] | None:
        if not profiles:
            return ()

        states: dict[
            tuple[int, int], tuple[tuple[float, float, int], tuple[_Candidate, ...]]
        ] = {(0, 0): ((0.0, 0.0, 0), ())}
        for profile in profiles:
            next_states: dict[
                tuple[int, int],
                tuple[tuple[float, float, int], tuple[_Candidate, ...]],
            ] = {}
            for (_, violations), (score, chosen) in states.items():
                for candidate in self._candidates(profile):
                    gpu_bytes = sum(item.gpu_bytes for item in chosen)
                    new_bytes = gpu_bytes + candidate.gpu_bytes
                    new_violations = violations + int(candidate.violates_slo)
                    if new_bytes > self.config.gpu_capacity_bytes:
                        continue
                    if new_violations > self.config.max_slo_violations:
                        continue
                    new_score = (
                        max(score[0], candidate.predicted_tbt_ms),
                        score[1] + candidate.predicted_tbt_ms,
                        new_bytes,
                    )
                    key = (new_bytes, new_violations)
                    previous = next_states.get(key)
                    if previous is None or new_score < previous[0]:
                        next_states[key] = (new_score, (*chosen, candidate))
            states = self._remove_dominated(next_states)
            if not states:
                return None

        return min(states.values(), key=lambda item: item[0])[1]

    def _candidates(self, profile: RequestProfile) -> tuple[_Candidate, ...]:
        candidates = []
        for num_gpu_layers in range(self.config.num_layers + 1):
            offloaded_layers = self.config.num_layers - num_gpu_layers
            predicted_tbt = (
                profile.compute_ms + offloaded_layers * profile.transfer_ms_per_layer
            )
            visible_tbt = max(0.0, predicted_tbt - profile.deposit_ms)
            candidates.append(
                _Candidate(
                    num_gpu_layers=num_gpu_layers,
                    gpu_bytes=num_gpu_layers * profile.kv_bytes_per_layer,
                    predicted_tbt_ms=predicted_tbt,
                    violates_slo=visible_tbt > profile.tbt_slo_ms,
                )
            )
        return tuple(candidates)

    @staticmethod
    def _remove_dominated(
        states: dict[
            tuple[int, int], tuple[tuple[float, float, int], tuple[_Candidate, ...]]
        ],
    ) -> dict[tuple[int, int], tuple[tuple[float, float, int], tuple[_Candidate, ...]]]:
        best: dict[
            tuple[int, int],
            tuple[tuple[float, float, int], tuple[_Candidate, ...]],
        ] = {}
        for key, value in states.items():
            previous = best.get(key)
            if previous is None or value[0] < previous[0]:
                best[key] = value
        return best

    def _to_placement(
        self, profile: RequestProfile, candidate: _Candidate
    ) -> RequestPlacement:
        gpu_layers = self._spread_layers(candidate.num_gpu_layers)
        return RequestPlacement(
            request_id=profile.request_id,
            gpu_layers=gpu_layers,
            predicted_tbt_ms=candidate.predicted_tbt_ms,
            gpu_bytes=candidate.gpu_bytes,
            violates_slo=candidate.violates_slo,
        )

    def _spread_layers(self, count: int) -> tuple[int, ...]:
        if count == 0:
            return ()
        if count == self.config.num_layers:
            return tuple(range(self.config.num_layers))
        return tuple(
            (index * self.config.num_layers) // count for index in range(count)
        )

    def _estimate_expiry(
        self,
        step: int,
        profiles: list[RequestProfile],
        placements: tuple[RequestPlacement, ...],
    ) -> int:
        expiry_deltas = []
        for profile, placement in zip(profiles, placements, strict=True):
            headroom = self.config.gpu_capacity_bytes - sum(
                item.gpu_bytes for item in placements
            )
            resident_layers = max(placement.num_gpu_layers, 1)
            growth_per_step = resident_layers * max(
                profile.kv_bytes_per_layer // max(step + 1, 1), 1
            )
            expiry_deltas.append(max(headroom // growth_per_step, 1))
        return step + min(expiry_deltas, default=1)
