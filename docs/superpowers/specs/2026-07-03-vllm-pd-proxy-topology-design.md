# vLLM PD Proxy and Topology Design

## Status

Approved for implementation planning.

## Context

`2026-07-03-pd-disaggregated-deployment-review.md` found that the current
remote PD topology support has one latent SGLang bug and one blocking vLLM
design bug:

- SGLang router does not pass each prefill worker's bootstrap port to
  `--prefill`, so deployments only work when every prefill and decode uses the
  same bootstrap port.
- vLLM PD command generation models multi-P/multi-D as one global
  `kv_parallel_size` and rank space. That does not match vLLM 0.23.0
  `P2pNcclConnector` or `NixlConnector`.
- vLLM PD also lacks a request proxy. vLLM workers expose producer/consumer
  roles, but the benchmark client still needs one OpenAI-compatible frontend
  that performs the prefill then decode request choreography.

This design replaces the string-template-first vLLM PD path with a
connector-aware topology model and adds a built-in proxy for the two supported
vLLM connectors.

## Goals

- Make SGLang PD router command generation robust when prefill bootstrap ports
  differ across nodes.
- Add first-class vLLM PD topology support for `P2pNcclConnector` and
  `NixlConnector`.
- Provide a built-in vLLM PD proxy that exposes `/v1/completions` and
  `/v1/chat/completions` to the benchmark runner.
- Keep existing single-service and existing SGLang PD configs compatible.
- Keep static multi-prefill and multi-decode topologies explicit; do not add
  dynamic discovery or automatic NIC/port detection.
- Validate unsupported or incomplete vLLM PD configurations before containers
  are launched.

## Non-Goals

- No Kubernetes, LWS, Helm, or service discovery.
- No automatic RDMA, NIC, IP, side-channel, or KV port discovery.
- No cross-host tensor-parallel worker group orchestration. Each
  `TopologyNode` still represents one container on one configured host.
- No GPU end-to-end validation in local unit tests. Local verification covers
  config rendering, proxy request construction, and lifecycle commands.
- No attempt to implement every vLLM connector. The supported set is
  `p2p_nccl` and `nixl`.

## Configuration Model

`topology_profiles[]` gains an optional `vllm_pd` object when
`engine == "vllm"` and `mode == "pd"`:

```json
{
  "name": "vllm_pd_p2p_2p2d",
  "engine": "vllm",
  "mode": "pd",
  "provider": "ssh_docker",
  "network": "host",
  "image": "vllm/vllm-openai:0.23.0",
  "vllm_pd": {
    "connector": "p2p_nccl",
    "proxy": {"kind": "builtin"},
    "p2p_send_type": "PUT_ASYNC",
    "nccl_num_channels": 16
  },
  "prefill": [
    {"name": "p1", "host": "p1", "port": 8100, "kv_port": 21001}
  ],
  "decode": [
    {"name": "d1", "host": "d1", "port": 8200, "kv_port": 22001}
  ],
  "frontend": {"kind": "builtin", "host": "proxy", "port": 8000}
}
```

For NIXL:

```json
{
  "vllm_pd": {
    "connector": "nixl",
    "proxy": {"kind": "builtin"}
  },
  "prefill": [
    {"name": "p1", "host": "p1", "port": 8100, "side_channel_port": 5601}
  ],
  "decode": [
    {"name": "d1", "host": "d1", "port": 8200, "side_channel_port": 5701}
  ]
}
```

The existing `kv_transfer_config_template` remains accepted for advanced
external-proxy use, but it is not the default path. When structured `vllm_pd`
is present, command rendering ignores `kv_transfer_config_template` and builds
known-good `KVTransferConfig` JSON.

## Topology Data

`TopologyNode` adds optional fields:

- `kv_port`: base KV transfer port used by vLLM P2P/NCCL.
- `side_channel_port`: NIXL side-channel port for this vLLM engine.

`TopologyFrontend.kind` accepts:

- `sglang_router`: existing SGLang router behavior.
- `external`: existing externally supplied command/image.
- `builtin`: built-in vLLM PD proxy launched from the bench-runner image.

For `frontend.kind == "builtin"`, the frontend image defaults to
`run.bench_image` unless `frontend.image` is set explicitly. This keeps the
proxy in the same package as benchmark utilities and avoids introducing a new
container image requirement.

## SGLang Fix

Router command generation changes from:

```text
--prefill http://<prefill-host>:<openai-port>
```

to:

```text
--prefill http://<prefill-host>:<openai-port> <bootstrap-port>
```

The bootstrap port comes from `TopologyNode.bootstrap_port`. In SGLang PD mode,
prefill nodes must have `bootstrap_port` set explicitly. Decode nodes may still
carry `bootstrap_port` for backward compatibility, but the router no longer
depends on decode-side fallback behavior.

## vLLM P2P/NCCL Worker Commands

For `vllm_pd.connector == "p2p_nccl"`:

- Prefill nodes render `kv_role=kv_producer`.
- Decode nodes render `kv_role=kv_consumer`.
- Every node must set `kv_port`.
- Generated `KVTransferConfig` includes:
  - `kv_connector: "P2pNcclConnector"`
  - `kv_role`
  - `kv_port`
  - `kv_connector_extra_config`

The generated config does not set `kv_rank` or `kv_parallel_size` for XPYD.
vLLM 0.23.0 P2P uses request-level pair selection, the per-worker distributed
rank, and remote addresses encoded in the request id; it is not a global
N-worker communicator.

The extra config includes stable fields that are useful to official examples
and do not change the static proxy contract:

```json
{
  "http_port": 8100,
  "send_type": "PUT_ASYNC",
  "nccl_num_channels": 16
}
```

Proxy discovery fields such as `proxy_ip` and `proxy_port` are not required for
the static built-in proxy. If later dynamic discovery is added, it should be a
separate proxy mode because it requires a registration transport and readiness
model.

## vLLM NIXL Worker Commands

For `vllm_pd.connector == "nixl"`:

- Prefill nodes render `kv_role=kv_producer`.
- Decode nodes render `kv_role=kv_consumer`.
- Every node must set `side_channel_port`.
- Generated `KVTransferConfig` includes:
  - `kv_connector: "NixlConnector"`
  - `kv_role`

Each node also receives environment variables:

```text
VLLM_NIXL_SIDE_CHANNEL_HOST=<node host address>
VLLM_NIXL_SIDE_CHANNEL_PORT=<node.side_channel_port>
```

The proxy receives normal HTTP endpoint addresses only. It obtains NIXL
transfer metadata from the prefill response and passes it to decode through
`kv_transfer_params`.

## Built-in Proxy

The proxy lives under `vllm_standalone_bench/vllm_bench/pd_proxy.py` and runs
as:

```text
python -m vllm_bench.pd_proxy --connector p2p_nccl --port 8000 ...
```

It uses `aiohttp.web`, which is already in `vllm_standalone_bench/requirements.txt`.

Endpoints:

- `GET /health`: local proxy health.
- `GET /v1`: compatibility readiness endpoint.
- `GET /v1/models`: forwarded to the first decode endpoint, falling back to
  the first prefill endpoint if decode does not respond.
- `POST /v1/completions`
- `POST /v1/chat/completions`

### P2P/NCCL Request Flow

1. Round-robin select one prefill node and one decode node.
2. Generate one request id:

   ```text
   ___prefill_addr_<prefill-kv-host>:<prefill-kv-port>___decode_addr_<decode-kv-host>:<decode-kv-port>_<uuid>
   ```

3. Send a non-streaming prefill request with `max_tokens=1`, the generated
   `request_id`, and compatibility headers `X-Request-Id` and `X-KV-Target`.
4. Send the original request body to decode with the same request id.
5. Stream the decode response back if the original request asked for streaming;
   otherwise return the decode JSON/body as-is.

### NIXL Request Flow

1. Round-robin select one prefill node and one decode node.
2. Send a non-streaming prefill request with `max_tokens=1` and:

   ```json
   {
     "kv_transfer_params": {
       "do_remote_decode": true,
       "do_remote_prefill": false,
       "remote_engine_id": null,
       "remote_block_ids": null,
       "remote_host": null,
       "remote_port": null
     }
   }
   ```

3. Read `kv_transfer_params` from the prefill response.
4. Send the original request body to decode with those params injected.
5. Return or stream the decode response.

## Validation Rules

- `engine == "vllm"` and `mode == "pd"` requires either structured `vllm_pd`
  or legacy `kv_transfer_config_template`.
- Structured `vllm_pd.connector` must be `p2p_nccl` or `nixl`.
- Built-in proxy only supports structured `vllm_pd`.
- P2P/NCCL requires `kv_port` on every prefill and decode node.
- NIXL requires `side_channel_port` on every prefill and decode node.
- SGLang PD requires `bootstrap_port` on every prefill node.
- Unknown `vllm_pd` keys fail fast instead of being silently ignored.
- Legacy template rendering must not inject nonstandard fields into structured
  mode.

## Compatibility

- Existing non-topology benchmark configs are unchanged.
- Existing SGLang PD configs continue to work if prefill bootstrap ports are
  present. Current examples already set them.
- Existing vLLM external proxy configs that use `kv_transfer_config_template`
  continue to render through the legacy path, but tests should stop presenting
  the invalid `kv_parallel_size=4` behavior as supported.
- New sample configs should prefer structured `vllm_pd`.

## Tests

Unit tests should be written before implementation changes and cover:

- SGLang router renders `--prefill <url> <bootstrap_port>`.
- SGLang prefill nodes require `bootstrap_port`.
- P2P/NCCL worker configs contain `P2pNcclConnector`, role, `kv_port`, and
  connector extras, and do not contain XPYD `kv_rank` or `kv_parallel_size`.
- NIXL worker configs contain `NixlConnector` and role, and node env contains
  side-channel host/port.
- Built-in proxy command renders one `--prefill` and `--decode` entry per
  node with HTTP endpoint and connector-specific KV metadata.
- Config parsing rejects missing `kv_port`, missing `side_channel_port`,
  unsupported connector names, and unknown structured `vllm_pd` keys.
- Proxy helpers generate P2P request ids in the exact format parsed by
  vLLM 0.23.0.
- Proxy helpers inject and forward NIXL `kv_transfer_params`.

Targeted verification:

```text
python -m pytest vllm_standalone_bench/tests/test_remote_topology.py -q
python -m pytest vllm_standalone_bench/tests/test_pd_proxy.py -q
python -m py_compile vllm_standalone_bench/vllm_bench/pd_proxy.py
git diff --check
```

## Open Constraints

- Local tests cannot prove GPU P2P/NCCL or NIXL transport correctness. That
  requires target hosts with the intended network and GPU topology.
- The static P2P proxy does not consume vLLM's optional proxy registration
  pings. It relies on configured HTTP and KV addresses, which matches this
  tool's static-topology design.
- Cross-host TP remains outside the topology model.
