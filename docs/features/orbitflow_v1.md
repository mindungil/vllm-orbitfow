# OrbitFlow on the V1 engine

OrbitFlow is available through the V1 KV connector and the native CPU
offloading backend. It computes request-wise layer residence under a GPU KV
budget, reuses a small ring of physical GPU staging blocks across offloaded
layers, and fences asynchronous CPU-to-GPU transfers at attention-layer
boundaries.

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --no-enable-prefix-caching \
  --kv-transfer-config '{
    "kv_connector":"OrbitFlowConnector",
    "kv_role":"kv_both",
    "kv_connector_extra_config":{
      "spec_name":"CPUOffloadingSpec",
      "cpu_bytes_to_use":8589934592,
      "eviction_policy":"lru",
      "num_resident_layers":8,
      "num_staging_banks":3,
      "prefetch_distance":2,
      "tbt_slo_ms":100.0,
      "layer_compute_ms":0.1,
      "transfer_bandwidth_gbps":24.0
    }
  }'
```

The V1 port reuses `OffloadingSpec`, HMA KV cache layouts, the shared V1 block
pool, and the connector lifecycle. Full-attention layers are split into
separate HMA cache groups so that each request can have a different resident
layer set. Resident groups own normal block-pool pages. Offloaded groups map
to request-disjoint pages in a shared staging ring, with pinned CPU tensors as
the durable backing store.

Loads are started ahead of the consuming attention layer and synchronized
immediately before its KV update. Stores run on a separate CUDA stream.
Residence changes use a two-step protocol: demoted layers are deposited and
synchronized first, then the scheduler atomically rewrites block ownership
and the worker loads promoted layers into their new pages.

The planner is deterministic and has no Gurobi dependency. Requests that
cannot fit the current residence budget are paused and reconsidered on the
next scheduler step.

Useful connector settings:

- `num_resident_layers`: physical layers with dedicated KV pages;
- `num_staging_banks`: ring banks reserved for offloaded layers;
- `prefetch_distance`: number of layers to prefetch ahead;
- `gpu_capacity_bytes`: planner admission/residence budget;
- `validate_transfers`: expensive byte-for-byte CPU/GPU validation for tests.

Current constraints are enforced at startup:

- one GPU (`tensor_parallel_size=1`, `pipeline_parallel_size=1`);
- full/dense attention layers with a uniform KV page size;
- no speculative decoding;
- prefix caching disabled.
- synchronous scheduling (the connector disables async scheduling);
- uniform full/dense-attention KV page sizes.

The port does not require a custom attention kernel, but it does require V1
core changes: per-layer HMA groups, allocator-level staging pages, a
pre-KV-update connector hook, and full block-table replacement after live
migration. Consequently it cannot be implemented as an out-of-tree connector
alone.
