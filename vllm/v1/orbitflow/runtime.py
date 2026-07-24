# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from vllm.v1.orbitflow.types import BatchPlacement


@dataclass(frozen=True, slots=True)
class LayerTransferAction:
    layer_index: int
    request_ids: tuple[str, ...]


class OrbitFlowLayerPipeline:
    """Coordinates layer-granular loads and stores around attention execution.

    The callbacks deliberately operate on request IDs and a layer index rather
    than backend-specific tensors. This keeps the placement policy independent
    from the CPU/NVMe offloading implementation used by the connector.
    """

    def __init__(
        self,
        layer_names: Iterable[str],
        *,
        prefetch_distance: int,
        load: Callable[[LayerTransferAction], None],
        wait: Callable[[LayerTransferAction], None],
        store: Callable[[LayerTransferAction], None],
    ) -> None:
        names = tuple(layer_names)
        if not names:
            raise ValueError("layer_names must not be empty")
        if len(set(names)) != len(names):
            raise ValueError("layer_names must be unique")
        if prefetch_distance < 0:
            raise ValueError("prefetch_distance must be non-negative")
        self._layer_names = names
        self._layer_indices = {name: i for i, name in enumerate(names)}
        self._prefetch_distance = prefetch_distance
        self._load = load
        self._wait = wait
        self._store = store
        self._actions: tuple[LayerTransferAction, ...] = ()
        self._submitted: set[int] = set()
        self._completed: set[int] = set()

    @property
    def layer_names(self) -> tuple[str, ...]:
        return self._layer_names

    def begin_batch(self, placement: BatchPlacement) -> None:
        request_layers = {
            request.request_id: frozenset(request.gpu_layers)
            for request in placement.requests
        }
        self._actions = tuple(
            LayerTransferAction(
                layer,
                tuple(
                    request_id
                    for request_id, resident_layers in request_layers.items()
                    if layer not in resident_layers
                ),
            )
            for layer in range(len(self._layer_names))
        )
        self._submitted.clear()
        self._completed.clear()
        self._prefetch_through(self._prefetch_distance)

    def wait_for_layer(self, layer_name: str) -> None:
        layer = self._layer_indices.get(layer_name)
        if layer is None:
            raise KeyError(f"unknown attention layer: {layer_name}")
        self._prefetch_through(layer + self._prefetch_distance)
        action = self._actions[layer]
        if action.request_ids and layer not in self._completed:
            self._wait(action)
            self._completed.add(layer)

    def store_layer(self, layer_name: str) -> None:
        layer = self._layer_indices.get(layer_name)
        if layer is None:
            raise KeyError(f"unknown attention layer: {layer_name}")
        action = self._actions[layer]
        if action.request_ids:
            self._store(action)

    def _prefetch_through(self, last_layer: int) -> None:
        if not self._actions:
            return
        last_layer = min(last_layer, len(self._actions) - 1)
        for layer in range(last_layer + 1):
            action = self._actions[layer]
            if action.request_ids and layer not in self._submitted:
                self._load(action)
                self._submitted.add(layer)
