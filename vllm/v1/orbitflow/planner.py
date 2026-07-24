# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
import time
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
    gpu_layers: tuple[int, ...]
    offloaded: tuple[bool, ...]
    gpu_bytes: int


class OrbitFlowPlanner:
    """Paper-style request-wise offload-distance MILP planner."""

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
        while selected is None and len(active) > 1:
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
        if selected is None and active:
            # As in OrbitFlow V0's distn_single fallback, never pause the
            # final runnable request solely because its TBT SLO is infeasible.
            # Otherwise no new profile or token can arrive to unblock it.
            selected = self._select_capacity_only(active)
        if selected is None and active:
            paused.append(active.pop().request_id)

        choices, token_time, window = selected or ((), 0.0, 1)
        placements = tuple(
            RequestPlacement(
                request_id=profile.request_id,
                gpu_layers=candidate.gpu_layers,
                predicted_tbt_ms=token_time,
                gpu_bytes=candidate.gpu_bytes,
                violates_slo=max(
                    0.0, token_time - profile.deposit_ms
                ) > profile.tbt_slo_ms,
            )
            for profile, candidate in zip(active, choices, strict=True)
        )
        return BatchPlacement(
            epoch=epoch,
            created_at_step=step,
            expires_at_step=step + window,
            reason=reason,
            requests=placements,
            paused_request_ids=tuple(paused),
        )

    def _select_capacity_only(
        self, profiles: list[RequestProfile]
    ) -> tuple[tuple[_Candidate, ...], float, int] | None:
        candidates = tuple(self._candidates(profile) for profile in profiles)
        feasible = (
            chosen
            for chosen in itertools.product(*candidates)
            if sum(candidate.gpu_bytes for candidate in chosen)
            <= self.config.gpu_capacity_bytes
        )
        chosen = min(
            feasible,
            key=lambda value: (
                self._token_time(profiles, value),
                sum(candidate.gpu_bytes for candidate in value),
            ),
            default=None,
        )
        if chosen is None:
            return None
        return (
            chosen,
            self._token_time(profiles, chosen),
            self._placement_window(profiles, chosen),
        )

    def _select(
        self, profiles: list[RequestProfile]
    ) -> tuple[tuple[_Candidate, ...], float, int] | None:
        if not profiles:
            return (), 0.0, 1
        candidates = tuple(self._candidates(profile) for profile in profiles)
        if self.config.solver_backend in {"auto", "gurobi"}:
            result = self._select_gurobi(profiles, candidates)
            if result is not None:
                return result
            if self.config.solver_backend == "gurobi":
                return None
        return self._select_search(profiles, candidates)

    def _select_gurobi(
        self,
        profiles: list[RequestProfile],
        candidates: tuple[tuple[_Candidate, ...], ...],
    ) -> tuple[tuple[_Candidate, ...], float, int] | None:
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError:
            return None

        model = gp.Model("orbitflow_v1")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = self.config.solver_timeout_ms / 1000
        x = {
            (r, c): model.addVar(vtype=GRB.BINARY, name=f"x_{r}_{c}")
            for r, options in enumerate(candidates)
            for c in range(len(options))
        }
        for r, options in enumerate(candidates):
            model.addConstr(gp.quicksum(x[r, c] for c in range(len(options))) == 1)
        model.addConstr(
            gp.quicksum(
                x[r, c] * candidate.gpu_bytes
                for r, options in enumerate(candidates)
                for c, candidate in enumerate(options)
            )
            <= self.config.gpu_capacity_bytes
        )
        window = model.addVar(
            lb=1,
            ub=self.config.max_plan_steps,
            vtype=GRB.INTEGER,
            name="decode_window",
        )
        selected_window = {}
        for r, options in enumerate(candidates):
            for c in range(len(options)):
                value = model.addVar(
                    lb=0,
                    ub=self.config.max_plan_steps,
                    name=f"selected_window_{r}_{c}",
                )
                selected_window[r, c] = value
                model.addConstr(value <= window)
                model.addConstr(value <= self.config.max_plan_steps * x[r, c])
                model.addConstr(
                    value
                    >= window
                    - self.config.max_plan_steps * (1 - x[r, c])
                )
        model.addConstr(
            gp.quicksum(
                x[r, c] * candidate.gpu_bytes
                + selected_window[r, c]
                * len(candidate.gpu_layers)
                * profiles[r].kv_growth_bytes_per_step
                for r, options in enumerate(candidates)
                for c, candidate in enumerate(options)
            )
            <= self.config.gpu_capacity_bytes
        )

        layers = self.config.num_layers
        compute_per_layer = max(p.compute_ms for p in profiles) / layers
        stalls = model.addVars(layers, lb=0.0, name="stall")
        for layer in range(layers):
            communication = gp.quicksum(
                x[r, c] * profiles[r].transfer_ms_per_layer
                for r, options in enumerate(candidates)
                for c, candidate in enumerate(options)
                if candidate.offloaded[layer]
            )
            if layer == 0:
                model.addConstr(stalls[layer] >= communication)
            else:
                model.addConstr(stalls[layer] >= communication - compute_per_layer)
        token_time = model.addVar(lb=0.0, name="token_time")
        model.addConstr(
            token_time
            == compute_per_layer * layers + gp.quicksum(stalls.values())
        )
        violation = model.addVars(len(profiles), vtype=GRB.BINARY, name="violation")
        big_m = max(
            sum(p.transfer_ms_per_layer for p in profiles) * layers
            + compute_per_layer * layers,
            1.0,
        )
        for r, profile in enumerate(profiles):
            model.addConstr(
                token_time - profile.deposit_ms - profile.tbt_slo_ms
                <= big_m * violation[r]
            )
        model.addConstr(
            gp.quicksum(violation.values()) <= self.config.max_slo_violations
        )
        model.setObjectiveN(token_time, 0, priority=2, weight=1)
        model.setObjectiveN(-window, 1, priority=1, weight=1)
        try:
            model.optimize()
        except gp.GurobiError:
            return None
        if model.SolCount == 0:
            return None
        chosen = tuple(
            options[
                max(range(len(options)), key=lambda c: x[r, c].X)
            ]
            for r, options in enumerate(candidates)
        )
        return chosen, float(token_time.X), max(int(window.X), 1)

    def _select_search(
        self,
        profiles: list[RequestProfile],
        candidates: tuple[tuple[_Candidate, ...], ...],
    ) -> tuple[tuple[_Candidate, ...], float, int] | None:
        deadline = time.perf_counter() + self.config.solver_timeout_ms / 1000
        best: tuple[tuple[float, int], tuple[_Candidate, ...]] | None = None
        for chosen in itertools.product(*candidates):
            if time.perf_counter() >= deadline and best is not None:
                break
            gpu_bytes = sum(candidate.gpu_bytes for candidate in chosen)
            if gpu_bytes > self.config.gpu_capacity_bytes:
                continue
            token_time = self._token_time(profiles, chosen)
            violations = sum(
                max(0.0, token_time - profile.deposit_ms)
                > profile.tbt_slo_ms
                for profile in profiles
            )
            if violations > self.config.max_slo_violations:
                continue
            score = (token_time, gpu_bytes)
            if best is None or score < best[0]:
                best = (score, chosen)
        if best is None:
            return None
        return (
            best[1],
            best[0][0],
            self._placement_window(profiles, best[1]),
        )

    def _token_time(
        self,
        profiles: list[RequestProfile],
        chosen: tuple[_Candidate, ...],
    ) -> float:
        layers = self.config.num_layers
        compute_per_layer = max(p.compute_ms for p in profiles) / layers
        stalls = 0.0
        for layer in range(layers):
            communication = sum(
                profile.transfer_ms_per_layer
                for profile, candidate in zip(profiles, chosen, strict=True)
                if candidate.offloaded[layer]
            )
            stalls += (
                communication
                if layer == 0
                else max(0.0, communication - compute_per_layer)
            )
        return compute_per_layer * layers + stalls

    def _candidates(self, profile: RequestProfile) -> tuple[_Candidate, ...]:
        layers = self.config.num_layers
        by_offload_count: dict[int, _Candidate] = {}
        for stride in range(1, layers + 2):
            offloaded = tuple(
                (layer + 1) % stride == 0 and stride <= layers
                for layer in range(layers)
            )
            gpu_layers = tuple(
                layer for layer, is_offloaded in enumerate(offloaded)
                if not is_offloaded
            )
            candidate = _Candidate(
                gpu_layers=gpu_layers,
                offloaded=offloaded,
                gpu_bytes=len(gpu_layers) * profile.kv_bytes_per_layer,
            )
            # The widest stride is the paper's representative for a given
            # offload count and provides the largest overlap window.
            by_offload_count[layers - len(gpu_layers)] = candidate
        return tuple(by_offload_count[count] for count in sorted(by_offload_count))

    def _placement_window(
        self,
        profiles: list[RequestProfile],
        chosen: tuple[_Candidate, ...],
    ) -> int:
        headroom = self.config.gpu_capacity_bytes - sum(
            candidate.gpu_bytes for candidate in chosen
        )
        growth = sum(
            len(candidate.gpu_layers) * profile.kv_growth_bytes_per_step
            for profile, candidate in zip(profiles, chosen, strict=True)
        )
        if growth <= 0:
            return self.config.max_plan_steps
        return max(
            min(
                int(headroom // growth),
                self.config.max_plan_steps,
            ),
            1,
        )
