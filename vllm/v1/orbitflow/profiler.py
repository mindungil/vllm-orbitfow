# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class OrbitFlowProfile:
    compute_ms: float = 0.0
    compute_layers: int = 0
    h2d_ms: float = 0.0
    h2d_bytes: int = 0
    d2h_ms: float = 0.0
    d2h_bytes: int = 0


@dataclass(slots=True)
class _TimedTransfer:
    start: torch.cuda.Event
    end: torch.cuda.Event
    num_bytes: int
    direction: str


class OrbitFlowRuntimeProfiler:
    """Collects completed CUDA timings without synchronizing the hot path."""

    def __init__(self) -> None:
        self._compute_starts: dict[str, torch.cuda.Event] = {}
        self._compute: list[_TimedTransfer] = []
        self._transfers: list[_TimedTransfer] = []

    def record_compute_start(self, layer_name: str, stream: torch.cuda.Stream) -> None:
        event = torch.cuda.Event(enable_timing=True)
        event.record(stream)
        self._compute_starts[layer_name] = event

    def record_compute_end(self, layer_name: str, stream: torch.cuda.Stream) -> None:
        start = self._compute_starts.pop(layer_name, None)
        if start is None:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record(stream)
        self._compute.append(_TimedTransfer(start, end, 0, "compute"))

    def start_transfer(self, stream: torch.cuda.Stream) -> torch.cuda.Event:
        event = torch.cuda.Event(enable_timing=True)
        event.record(stream)
        return event

    def end_transfer(
        self,
        start: torch.cuda.Event,
        stream: torch.cuda.Stream,
        num_bytes: int,
        direction: str,
    ) -> None:
        end = torch.cuda.Event(enable_timing=True)
        end.record(stream)
        self._transfers.append(
            _TimedTransfer(start, end, num_bytes, direction)
        )

    def collect(self) -> OrbitFlowProfile:
        compute_ms = 0.0
        compute_layers = 0
        pending_compute = []
        for timing in self._compute:
            if not timing.end.query():
                pending_compute.append(timing)
                continue
            compute_ms += timing.start.elapsed_time(timing.end)
            compute_layers += 1
        self._compute = pending_compute

        h2d_ms = d2h_ms = 0.0
        h2d_bytes = d2h_bytes = 0
        pending_transfers = []
        for timing in self._transfers:
            if not timing.end.query():
                pending_transfers.append(timing)
                continue
            elapsed = timing.start.elapsed_time(timing.end)
            if timing.direction == "h2d":
                h2d_ms += elapsed
                h2d_bytes += timing.num_bytes
            else:
                d2h_ms += elapsed
                d2h_bytes += timing.num_bytes
        self._transfers = pending_transfers
        return OrbitFlowProfile(
            compute_ms=compute_ms,
            compute_layers=compute_layers,
            h2d_ms=h2d_ms,
            h2d_bytes=h2d_bytes,
            d2h_ms=d2h_ms,
            d2h_bytes=d2h_bytes,
        )
