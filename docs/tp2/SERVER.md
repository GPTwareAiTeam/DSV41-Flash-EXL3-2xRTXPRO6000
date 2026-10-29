# TP2 OpenAI-compatible server

The server runs one process per GPU, with rank 0 serving HTTP and rank 1 participating in
inference. It needs a TP=2 sharded checkpoint and Engram tables as described in
[RECIPE.md](RECIPE.md). The launcher expects an **already running**, clean container named
`exl3-server` with both GPUs, the installed Python dependencies and CUDA toolkit, the repository,
and model files available inside it. It does not create or modify the container. By default the
repository is at `/src/exllamav3` and the overlay is at `/artifacts/tp2-overlay` in the container.

Use host networking to reach the default loopback listener from the host. With Docker bridge
networking, binding only container loopback will not expose the service through a published
port; set `HOST=0.0.0.0` and publish the port to host loopback instead (for example,
`-p 127.0.0.1:8000:8000` when creating the container). A non-loopback listener can expose the
API to other machines; protect it appropriately.

## Start and manage

```sh
# Run from the repository on the host. These paths are container paths.
qmesh/tp2/serve_tp2.sh start -e OVERLAY=/artifacts/tp2-overlay
qmesh/tp2/serve_tp2.sh status
qmesh/tp2/serve_tp2.sh stop
```

`start` launches two detached `docker exec` processes using `python -m qmesh.tp2.serve_tp2`; it returns
before the model is loaded. `status` checks whether both rank processes are alive, not whether
the model is ready, and exits nonzero if either is stopped. `stop` sends SIGTERM to rank 0,
which coordinates shutdown with rank 1. Each launch creates a fresh rendezvous and P2P directory under `STATE_DIR`, avoiding stale
handshakes. Rank logs are in the printed `run.XXXXXXXX/rank0.log` and `rank1.log` paths inside
the container; inspect them with `docker exec exl3-server cat <path>`. A running rank prevents
another start; stop both ranks first.

Launcher settings can be set in the host environment:

| Variable | Default | Purpose |
|---|---|---|
| `CONTAINER` | `exl3-server` | Existing running container. |
| `REPO_DIR` | `/src/exllamav3` | Working directory inside the container. |
| `STATE_DIR` | `/tmp/qmesh-tp2-server` | Container directory for PID files, rendezvous and logs. Use the same value for start, status and stop. |
| `QMESH_CPUSET_RANK0` | `2-5` | Rank 0 CPU affinity list; empty disables pinning. |
| `QMESH_CPUSET_RANK1` | `10-13` | Rank 1 CPU affinity list; empty disables pinning. |
| `HOST` | `127.0.0.1` | HTTP bind address inside the container. |
| `PORT` | `8000` | HTTP port. |
| `DSPARK` | `1` | Enable speculative decoding; set `0` for plain decoding. |
| `NUM_DRAFT` | `5` | Draft tokens per speculative round. |
| `BATCH` | `1` | Number of cache slots and maximum batch size. |
| `MODEL_ID` | `deepseek-v4.1-flash` | Model identifier exposed by the API. |
| `API_KEY` | unset | Optional bearer key required on all endpoints when set. |

The launcher applies each rank's affinity with `taskset -c` inside the container. The defaults
are distinct physical cores on NUMA node 0 on the target host; override the lists for other
hosts or set either variable to an empty string to leave that rank unpinned.

Exported model/runtime settings including `OVERLAY`, `COMM`, `CTX_TOKENS`, `BATCH`,
`MODEL_ID`, `API_KEY`, `EXL3_*`, `DSPARK_*`, `TORCH_*` and `CUDA_VISIBLE_DEVICES` are
forwarded to both ranks. Other settings can be passed explicitly with repeated Docker-style
`-e KEY=VALUE` options after `start`; these take precedence over forwarded values. The launcher
sets `RANK`, `WORLD_SIZE`, `INIT_METHOD` and `P2P_RDV` itself. See
[env_vars.md](env_vars.md) for the runtime and memory configuration options.

## API

The base URL with the defaults and host networking is `http://127.0.0.1:8000/v1`.
The server exposes:

- `GET /v1/models` — list the served model.
- `POST /v1/chat/completions` — chat completion.
- `POST /v1/completions` — text completion.

The API supports streaming via server-sent events (`stream: true`) and `n=1`. Generation
options include `max_tokens` (or `max_completion_tokens`), `temperature`, `top_p`,
`presence_penalty`, `frequency_penalty`, `stop` and `seed`; chat requests also accept
`tools`. Set `stream_options.include_usage` to request usage in a streaming response.
`tools` is accepted on chat requests only; `/v1/completions` rejects it with HTTP 400.
`finish_reason` is `stop` for an end-of-sequence or stop-string stop, `length` when
`max_tokens` was reached and `tool_calls` when the model emitted tool calls. Each non-streaming
choice also carries a non-standard `token_ids` array with the generated token IDs, which is
handy for exact-output comparisons against `run_tp2.py`.
Obtain the model ID from `/v1/models`, then pass it as `model` in a request:

```sh
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"Hello!"}],"max_tokens":64}'
```

When `API_KEY` is set, send `Authorization: Bearer <key>` on every endpoint. Invalid
requests return HTTP 400, authentication failures 401, and engine failures 500 (or an
error event if streaming has already begun). Without a key the API is unauthenticated;
restrict access to the listener.

Validation happens before a request is queued: an unknown `model`, `n > 1`, a non-positive
`max_tokens`, more than four `stop` strings, an empty prompt, or a prompt plus completion
that does not fit the KV cache returns HTTP 400 and leaves the engine untouched. Requests
arriving while a batch is generating are queued and admitted as a new round, so concurrent
clients share the server but never join a round already in progress; with `BATCH=1` they
are served one after another.

`usage` reports `prompt_tokens`, `completion_tokens` and `total_tokens`, plus
`prompt_tokens_details.cached_tokens` for tokens reused from the prefix cache. A repeated
prefix therefore shows up as a nonzero `cached_tokens` with unchanged output.

## Validation

`scripts/server_smoke.sh` runs the server end to end inside a throwaway container created from
the clean image, on a private port and extension-cache directory: it first produces a reference
token sequence with `run_tp2.py` on the same prompt, then checks the endpoint set, exact token
identity with that reference, streaming text and usage, prefix-cache reuse on a repeated long
prefix, tool-call handling (and its rejection on `/v1/completions`), and finally that `stop`
releases both rank processes. Run it through `gpu_run.sh` so the GPU block is accounted for.
