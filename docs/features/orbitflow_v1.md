# OrbitFlow on the V1 engine

OrbitFlow is available as a V1 KV connector. It computes request-wise layer
residence under a GPU KV budget, reuses a small ring of physical GPU staging
blocks across offloaded layers, and fences asynchronous CPU-to-GPU transfers
at attention-layer boundaries.

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --no-enable-prefix-caching \
  --kv-transfer-config '{
    "kv_connector":"OrbitFlowConnector",
    "kv_role":"kv_both",
    "kv_connector_extra_config":{
      "cpu_cache_bytes":8589934592,
      "num_resident_layers":8,
      "num_staging_banks":3,
      "prefetch_distance":2,
      "solver_backend":"auto",
      "solver_timeout_ms":20,
      "profile_ewma_alpha":0.2,
      "tbt_slo_ms":100.0,
      "layer_compute_ms":0.1,
      "transfer_bandwidth_gbps":24.0
    }
  }'
```

The V1 port reuses HMA KV cache layouts, the shared V1 block pool, and the
connector lifecycle. Full-attention layers are split into separate HMA cache
groups so that each request can have a different resident layer set. Resident
groups own normal block-pool pages. Offloaded groups map to request-disjoint
pages in a shared staging ring, with pinned CPU tensors as the durable backing
store. OrbitFlow owns these layer transfers directly: the generic V1
offloading job tracker cannot be used because it assumes block IDs are unique
across groups, while OrbitFlow deliberately reuses staging IDs across layers.

Loads are started ahead of the consuming attention layer and synchronized
immediately before its KV update. Stores run on a separate CUDA stream.
Residence changes use a two-step protocol: demoted layers are deposited and
synchronized first, every tensor-parallel worker acknowledges completion,
then the scheduler atomically rewrites block ownership and the worker loads
promoted layers into their new pages.

The planner models the paper's per-request offload-distance candidates,
per-layer communication flow and stalls, GPU capacity, Token-Deposit latency
margin, SLO violation budget, and placement decode window. `solver_backend`
may be `gurobi`, `search`, or `auto`. Gurobi is optional; `auto` uses it when
installed and otherwise searches the same discrete candidate set under
`solver_timeout_ms`. Requests that cannot fit are paused and reconsidered on
the next scheduler step. The last runnable request falls back to the
lowest-latency capacity-feasible placement instead of being paused forever.

CUDA events measure attention-layer compute and H2D/D2H transfers. Worker
measurements are aggregated across tensor-parallel ranks and returned through
`KVConnectorWorkerMetadata`; the scheduler feeds their EWMA back into the
next optimization epoch.

Pinned backing is allocated as a contiguous request-layer arena. Consecutive
physical pages are transferred as one DMA run instead of one Python call per
block. An optional NVMe tier spills least-recently-used arenas when the pinned
CPU budget is exceeded:

```json
{
  "cpu_cache_bytes": 8589934592,
  "nvme_path": "/mnt/nvme/orbitflow",
  "nvme_bytes": 137438953472
}
```

NVMe promotion synchronizes a pending DMA before evicting its pinned arena.
This preserves correctness but makes NVMe a capacity tier, not a latency
equivalent replacement for pinned CPU memory.

Useful connector settings:

- `num_resident_layers`: physical layers with dedicated KV pages;
- `num_staging_banks`: ring banks reserved for offloaded layers;
- `prefetch_distance`: number of layers to prefetch ahead;
- `gpu_capacity_bytes`: planner admission/residence budget;
- `validate_transfers`: debug-only byte-for-byte CPU/GPU validation; it
  synchronizes and scales with the number of blocks;
- `solver_backend`: `auto`, `gurobi`, or exact-search fallback;
- `profile_ewma_alpha`: runtime-profile feedback smoothing;
- `cpu_cache_bytes`, `nvme_path`, `nvme_bytes`: CPU/NVMe tier budgets.

`cpu_bytes_to_use` and `cpu_bytes_to_use_per_rank` remain accepted as
compatibility aliases for `cpu_cache_bytes`.

Current constraints are enforced at startup:

- tensor parallelism is supported; pipeline parallelism is not;
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
