#!/usr/bin/env python3
"""Run the launch-merge identity matrix, then profile baseline and all dials.

Both ranks must generate 128 identical tokens for every prompt in all identity
arms, including the 24k-token context, before any profiling arm is launched.
Run only when other GPU launcher users agree to wait until this script exits.
The launcher takes gpu.lock itself; do not wrap this script in that lock.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

ARTIFACTS = Path('/home/treadall/recipe-artifacts/2x-rtx-tp2')
LOGS = ARTIFACTS / 'tp2logs'
CONTAINER = 'exl3-tp2-lm'
DIALS = ('EXL3_DSV41_FUSED_ROPE', 'EXL3_DSV41_FUSED_POOL_SCATTER',
         'EXL3_DSV4_FUSED_INDEXER_ROPE')
ARMS = {'off': (0, 0, 0), 'rope': (1, 0, 0), 'scatter': (0, 1, 0),
        'indexer': (0, 0, 1), 'all': (1, 1, 1)}
# Check process environments, not command-line tags: bash may exec Python.
PROBE = r'''import os
needle = os.environ['CHECK_TOKENS_OUT'].encode()
for name in os.listdir('/proc'):
    if not name.isdigit(): continue
    try:
        env = open('/proc/' + name + '/environ', 'rb').read().split(b'\0')
        cmd = open('/proc/' + name + '/cmdline', 'rb').read()
        if b'TOKENS_OUT=' + needle in env and b'qmesh.tp2.run_tp2' in cmd:
            print(name)
    except (OSError, ProcessLookupError): pass
'''


def command(args, timeout, **kwargs):
    return subprocess.run(args, timeout=timeout, check=True, **kwargs)


def ranks_running(tag):
    out = command(['docker', 'exec', '-e', f'CHECK_TOKENS_OUT=/artifacts/tp2logs/{tag}_tokens',
                   CONTAINER, 'python3', '-c', PROBE], 30, capture_output=True, text=True)
    return out.stdout.split()


def run_arm(tag, values, long_context, profile, timeout):
    options = dict(zip(DIALS, map(str, values)))
    options.update({'PROMPT_IDS': '0' if long_context else '0,1,2',
                    'LONG_PROMPT_TOKENS': '20000' if long_context else '0',
                    'CTX_TOKENS': '65536', 'PROFILE': str(int(profile)),
                    'PROFILE_SEQ': '', 'BATCH': '1', 'MAX_NEW': '128',
                    'IGNORE_EOS': '1', 'COMM': 'p2p', 'TORCH_CUDA_ARCH_LIST': '12.0a'})
    args = [str(ARTIFACTS / 'run_lm_model.sh'), tag]
    for key, val in options.items():
        args.extend(['-e', f'{key}={val}'])
    telemetry = None
    try:
        gpu_file = open(LOGS / f'{tag}_gpu.csv', 'x')
        try:
            telemetry = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,memory.used,utilization.gpu,clocks.sm,power.draw,clocks_throttle_reasons.active',
                                          '--format=csv,noheader', '-l', '1'], stdout=gpu_file, stderr=subprocess.STDOUT)
        finally:
            gpu_file.close()
        print(f'Launching {tag}: {options}', flush=True)
        deadline = time.monotonic() + timeout
        with open(LOGS / f'{tag}_launcher.log', 'x') as log:
            command(args, timeout, cwd=ARTIFACTS, stdout=log, stderr=subprocess.STDOUT)
        while ranks_running(tag):
            if time.monotonic() >= deadline:
                raise TimeoutError(f'{tag}: rank processes still running; no further arms launched')
            time.sleep(2)
        for rank in (0, 1):
            log = (LOGS / f'{tag}_rank{rank}.log').read_text(errors='replace')
            if 'Traceback (most recent call last)' in log or 'decode_ms_per_tok=' not in log:
                raise RuntimeError(f'{tag}: rank {rank} failed or lacks timing; inspect log')
        if profile and (not (LOGS / f'{tag}_trace_rank0.json').is_file() or
                        (LOGS / f'{tag}_trace_rank0.json').stat().st_size == 0):
            raise RuntimeError(f'{tag}: missing or empty profile trace')
        if telemetry and (telemetry.poll() is not None or
                          (LOGS / f'{tag}_gpu.csv').stat().st_size == 0):
            raise RuntimeError(f'{tag}: GPU telemetry exited or produced no samples')
    finally:
        if telemetry and telemetry.poll() is None:
            telemetry.terminate()
            try:
                telemetry.wait(timeout=10)
            except subprocess.TimeoutExpired:
                telemetry.kill()
                telemetry.wait()


def load_tokens(tag, expected):
    result = []
    for rank in (0, 1):
        path = LOGS / f'{tag}_tokens.rank{rank}.json'
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or set(data) != expected or any(
            not isinstance(v, list) or len(v) != 128 or any(type(t) is not int for t in v)
            for v in data.values()
        ):
            raise ValueError(f'{path}: missing prompt or invalid token sequence (expected 128 integers)')
        result.append(data)
    return result


def report_arm(tag, profile):
    for rank in (0, 1):
        path = LOGS / f'{tag}_rank{rank}.log'
        lines = [line for line in path.read_text(errors='replace').splitlines()
                 if 'decode_ms_per_tok=' in line]
        print(f'{tag} rank{rank} timings ({path}):', flush=True)
        for line in lines:
            print(line, flush=True)
    print(f'{tag} telemetry: {LOGS / (tag + "_gpu.csv")}', flush=True)
    if profile:
        rank0 = (LOGS / f'{tag}_rank0.log').read_text(errors='replace')
        steps = re.findall(r'PROFILE decode_steps=(\d+)\b', rank0)
        if len(steps) != 1 or int(steps[0]) <= 0:
            raise ValueError(f'{tag}: missing, ambiguous or zero profile decode_steps')
        trace_path = LOGS / f'{tag}_trace_rank0.json'
        trace = json.loads(trace_path.read_text())
        events = trace.get('traceEvents') if isinstance(trace, dict) else None
        if not isinstance(events, list):
            raise ValueError(f'{trace_path}: missing traceEvents')
        categories = {'cudaLaunchKernel': 'cuda_runtime',
                      'cudaLaunchKernelExC': 'cuda_runtime',
                      'cuLaunchKernelEx': 'cuda_driver',
                      'cudaLaunchCooperativeKernel': 'cuda_runtime',
                      'cudaGraphLaunch': 'cuda_runtime'}
        counts = {name: sum(1 for event in events if isinstance(event, dict)
                            and event.get('ph') == 'X' and event.get('cat') == category
                            and event.get('name') == name)
                  for name, category in categories.items()}
        if not counts['cudaLaunchKernel']:
            raise ValueError(f'{trace_path}: no cudaLaunchKernel runtime events')
        n = int(steps[0])
        print(f'{tag} rank0 trace: {trace_path}; decode_steps={n}; '
              + ', '.join(f'{name}={count} ({count / n:.3f}/step)'
                          for name, count in counts.items()), flush=True)
        for line in rank0.splitlines():
            if 'PROFILE decode_steps=' in line:
                print(f'{tag} {line}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefix', help='output tag prefix (letters, digits, underscores, hyphens)')
    parser.add_argument('--timeout', type=int, default=5400, help='maximum seconds per arm, including launcher lock wait')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.prefix) or args.timeout <= 0:
        parser.error('invalid prefix or timeout')
    session = f'{args.prefix}_{time.time_ns()}_{uuid.uuid4().hex[:8]}'
    # Separate from gpu.lock, which is acquired by run_lm_model.sh.
    with open(ARTIFACTS / 'compare_launch_merge.lock', 'w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('another comparison harness is running')
        print(f'Session {session}; coordinate with all other launcher users until completion.', flush=True)
        references = {}
        for context, expected in [('short', {'0', '1', '2'}), ('long', {'0'})]:
            for arm, values in ARMS.items():
                tag = f'{session}_identity_{context}_{arm}'
                run_arm(tag, values, context == 'long', False, args.timeout)
                results = load_tokens(tag, expected)
                if context not in references:
                    references[context] = results[0]
                if any(result != references[context] for result in results):
                    raise RuntimeError(f'{context} {arm}: tokens differ across arms or ranks; profiling blocked')
                report_arm(tag, False)
            print(f'{context}: both ranks and all arms have identical tokens', flush=True)
        for context in ('short', 'long'):
            for arm in ('off', 'all'):
                tag = f'{session}_profile_{context}_{arm}'
                run_arm(tag, ARMS[arm], context == 'long', True, args.timeout)
                results = load_tokens(tag, {'0'} if context == 'long' else {'0', '1', '2'})
                if any(result != references[context] for result in results):
                    raise RuntimeError(f'{context} {arm}: profile tokens differ from identity baseline')
                report_arm(tag, True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, TimeoutError, subprocess.SubprocessError) as exc:
        print(f'ERROR: {exc}. No further arms will be launched; check for surviving ranks before retrying.', file=sys.stderr)
        sys.exit(1)
