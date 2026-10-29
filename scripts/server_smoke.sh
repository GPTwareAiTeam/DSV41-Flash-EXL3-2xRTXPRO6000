#!/usr/bin/env bash
# Run with bash scripts/server_smoke.sh. GPU access is serialized by gpu_run.sh.
set -euo pipefail
A=${ARTIFACTS_DIR:-/home/treadall/recipe-artifacts/2x-rtx-tp2}
SRC=${SRC_DIR:-/home/treadall/server-src}
if [[ ${1:-} != --queued ]]; then
  exec "$A/gpu_run.sh" Server "${GPU_MINUTES:-45}" bash "$0" --queued
fi
D=$(mktemp -d "$A/tp2logs/server-smoke.XXXXXXXX")
inside="/artifacts/tp2logs/${D##*/}"
C="exl3-server-smoke-$$"
PORT=${PORT:-18080}
EXT=${EXT_DIR:-/mnt/data12/work/exl3/ext_server}
IMAGE=${IMAGE:-exl3-tp2:clean-20261028}
[[ -s "$EXT/exllamav3_ext/exllamav3_ext.so" ]] || { echo "Missing clean extension in $EXT" >&2; exit 1; }
cleanup() { docker stop -t 20 "$C" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM
[[ -z ${GPU_RUN_STATUS:-} ]] || echo 'eta_min=30 doing=server_smoke_reference' > "$GPU_RUN_STATUS"
docker run -d --init --name "$C" --gpus '"device=0,1"' --network host --ipc host --shm-size 64g --security-opt label=disable \
  -e CUDA_HOME=/opt/cuda -e TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions \
  -v /models:/models:ro -v /usr/local/cuda:/opt/cuda:ro -v "$SRC":/src/exllamav3:ro \
  -v "$EXT":/root/.cache/torch_extensions -v "$A":/artifacts \
  --entrypoint sleep "$IMAGE" infinity >/dev/null
pids=()
for rank in 0 1; do
  timeout -k 10 1500 docker exec -e RANK="$rank" -e WORLD_SIZE=2 \
    -e INIT_METHOD="file://$inside/reference-rdv" -e P2P_RDV="$inside/reference-p2p" \
    -e NCCL_SOCKET_IFNAME=lo -e COMM=p2p -e DSPARK=1 -e NUM_DRAFT=5 \
    -e MAX_NEW=16 -e PROMPT_IDS=0 -e TOKENS_OUT="$inside/reference" \
    "$C" bash -c 'cd /src/exllamav3 && exec python3 -u -m qmesh.tp2.run_tp2' > "$D/reference-rank$rank.log" 2>&1 &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
(( rc == 0 )) || { echo "Reference failed; logs in $D" >&2; exit 1; }
[[ -z ${GPU_RUN_STATUS:-} ]] || echo 'eta_min=20 doing=server_smoke_endpoints' > "$GPU_RUN_STATUS"
CONTAINER="$C" REPO_DIR=/src/exllamav3 bash "$SRC/qmesh/tp2/serve_tp2.sh" start \
  -e HOST=127.0.0.1 -e PORT="$PORT" -e DSPARK=1 -e NUM_DRAFT=5 -e NCCL_SOCKET_IFNAME=lo \
  -e EXL3_PREFIX_CACHE=1 -e EXL3_PREFIX_CACHE_INTERVAL=256 -e EXL3_PREFIX_CACHE_MIN_MATCH=256
python3 - "$PORT" "$D" <<'PY'
import json, pathlib, sys, time, urllib.error, urllib.request
port, directory = int(sys.argv[1]), pathlib.Path(sys.argv[2])
base = f'http://127.0.0.1:{port}'
def request(path, body=None, timeout=120):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, headers={'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=timeout)
for attempt in range(180):
    try:
        with request('/v1/models', timeout=2) as response:
            models = json.load(response)['data']
        break
    except (OSError, urllib.error.URLError):
        time.sleep(2)
else:
    raise RuntimeError('server did not become ready')
assert models and models[0]['id'], models
model = models[0]['id']
with request('/v1/completions', {'model': model, 'prompt': 'The capital of France is',
                                'temperature': 0, 'max_tokens': 8}) as response:
    completion = json.load(response)
assert completion['choices'][0]['text'] and completion['usage']['completion_tokens'] > 0, completion
def chat(content, **kwargs):
    body = {'model': model, 'messages': [{'role': 'user', 'content': content}],
            'temperature': 0, 'max_tokens': 16, **kwargs}
    return request('/v1/chat/completions', body)
prompt = 'The capital of France is'
with chat(prompt) as response:
    result = json.load(response)
refs = [json.loads((directory / f'reference.rank{rank}.json').read_text())['0'] for rank in (0, 1)]
assert refs[0] == refs[1], 'reference ranks disagree'
assert result['choices'][0]['token_ids'] == refs[0], (result['choices'][0]['token_ids'], refs[0])
assert result['usage']['completion_tokens'] > 0
with chat(prompt, stream=True, stream_options={'include_usage': True}) as response:
    events, done = [], False
    for raw in response:
        line = raw.decode().strip()
        if not line.startswith('data:'):
            continue
        data = line[5:].strip()
        if data == '[DONE]':
            done = True
            break
        events.append(json.loads(data))
assert done and events, 'missing streaming events or [DONE]'
text = ''.join(choice.get('delta', {}).get('content') or '' for event in events for choice in event.get('choices', []))
assert text == result['choices'][0]['message']['content'], (text, result['choices'][0]['message']['content'])
assert any(event.get('usage') for event in events), 'missing streaming usage'
long_prompt = ('The quick brown fox jumps over the lazy dog. ' * 400) + 'Summarize this in one sentence.'
with chat(long_prompt) as response:
    first = json.load(response)
messages = [{'role': 'user', 'content': long_prompt},
            {'role': 'assistant', 'content': first['choices'][0]['message']['content']},
            {'role': 'user', 'content': 'What animal was mentioned?'}]
with request('/v1/chat/completions', {'model': model, 'messages': messages,
                                      'temperature': 0, 'max_tokens': 16}) as response:
    second = json.load(response)
for item in (first, second):
    usage = item['usage']
    assert usage['prompt_tokens'] > 256 and usage['completion_tokens'] > 0, usage
cached = second['usage']['prompt_tokens_details']['cached_tokens']
assert second['choices'][0]['token_ids'], 'second turn produced no tokens'
tools = [{'type': 'function', 'function': {'name': 'get_weather',
          'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}]
with chat('What is the weather in Paris? Use the tool.', tools=tools, max_tokens=48) as response:
    tooled = json.load(response)
assert tooled['choices'][0]['finish_reason'] in ('tool_calls', 'stop'), tooled
assert tooled['choices'][0]['message']['role'] == 'assistant', tooled
try:
    request('/v1/completions', {'model': model, 'prompt': 'hi', 'tools': tools, 'max_tokens': 4})
except urllib.error.HTTPError as exc:
    assert exc.code == 400, exc.code
else:
    raise AssertionError('tools must be rejected on /v1/completions')
assert cached > 0, second['usage']
print(f'PASS: model={model}, reference tokens={len(refs[0])}, cached_tokens={cached}, streaming complete')
PY
CONTAINER="$C" REPO_DIR=/src/exllamav3 bash "$SRC/qmesh/tp2/serve_tp2.sh" stop
for attempt in {1..60}; do
  if docker exec "$C" sh -c 'for r in 0 1; do f=/tmp/qmesh-tp2-server/rank$r.pid; [ ! -r "$f" ] || ! kill -0 "$(cat "$f")" 2>/dev/null || exit 1; done' \
     && ! (exec 3<>/dev/tcp/127.0.0.1/"$PORT") 2>/dev/null; then
    if docker exec "$C" sh -c 'grep -q Traceback /tmp/qmesh-tp2-server/run.*/rank*.log'; then
      echo 'rank log has a traceback' >&2; exit 1
    fi
    printf 'PASS: shutdown; logs=%s\n' "$D"
    exit 0
  fi
  sleep 1
done
echo 'server ranks did not exit after stop' >&2
exit 1
