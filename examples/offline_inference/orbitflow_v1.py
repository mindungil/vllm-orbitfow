# SPDX-License-Identifier: Apache-2.0
"""Minimal single-GPU OrbitFlow V1 example."""

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig


def main() -> None:
    kv_transfer_config = KVTransferConfig(
        kv_connector="OrbitFlowConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "spec_name": "CPUOffloadingSpec",
            "cpu_bytes_to_use": 8 << 30,
            "eviction_policy": "lru",
            "num_resident_layers": 8,
            "num_staging_banks": 3,
            "prefetch_distance": 2,
            "tbt_slo_ms": 100.0,
            "layer_compute_ms": 0.1,
            "transfer_bandwidth_gbps": 24.0,
        },
    )
    llm = LLM(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        max_model_len=2048,
        tensor_parallel_size=1,
        enable_prefix_caching=False,
        kv_transfer_config=kv_transfer_config,
    )
    outputs = llm.generate(
        ["The capital of France is"],
        SamplingParams(temperature=0, max_tokens=16),
    )
    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
