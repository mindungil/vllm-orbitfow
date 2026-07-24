# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config import VllmConfig

ORBITFLOW_CONNECTOR_NAME = "OrbitFlowConnector"


def is_orbitflow_enabled(vllm_config: "VllmConfig") -> bool:
    transfer_config = vllm_config.kv_transfer_config
    return bool(
        transfer_config and transfer_config.kv_connector == ORBITFLOW_CONNECTOR_NAME
    )


def get_orbitflow_extra_config(vllm_config: "VllmConfig") -> dict[str, Any]:
    if not is_orbitflow_enabled(vllm_config):
        return {}
    transfer_config = vllm_config.kv_transfer_config
    assert transfer_config is not None
    return dict(transfer_config.kv_connector_extra_config or {})
