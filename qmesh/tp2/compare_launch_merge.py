#!/usr/bin/env python3
"""Compare one 0/1 environment dial using the shared TP2 model launcher.

Run only when other users of the GPU launcher have agreed not to launch until
this entire script exits. The launcher takes gpu.lock itself; do not wrap this
script in that lock. See --help for the command line.
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


def run_arm(tag, dial, value, long_context, profile, timeout):
    options = {dial: str(value), 'PROMPT_IDS': '0' if long_context else '0,1,2',
               'LONG_PROMPT_TOKENS': '20000' if long_context else '0',
               'CTX_TOKENS': '65536', 'PROFILE': str(int(profile)), 'BATCH': '1',
               'TORCH_CUDA_ARCH_LIST': '12.0a'}
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
            not isinstance(v, list) or not v or any(type(t) is not int for t in v)
            for v in data.values()
        ):
            raise ValueError(f'{path}: missing prompt or invalid/empty tokens')
        result.append(data)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dial', help='environment variable to compare at 0 and 1')
    parser.add_argument('prefix', help='output tag prefix (letters, digits, underscores, hyphens)')
    parser.add_argument('--timeout', type=int, default=5400, help='maximum seconds per arm, including launcher lock wait')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', args.dial) or args.dial in {
        'PROMPT_IDS', 'LONG_PROMPT_TOKENS', 'CTX_TOKENS', 'PROFILE', 'PROFILE_SEQ', 'BATCH',
        'TOKENS_OUT', 'PROFILE_TRACE', 'RANK', 'WORLD_SIZE', 'INIT_METHOD', 'P2P_RDV'}:
        parser.error('dial must be an unreserved environment variable name')
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
        for context, expected in [('short', {'0', '1', '2'}), ('long', {'0'})]:
            results = []
            for value in (0, 1):
                tag = f'{session}_identity_{context}_{value}'
                run_arm(tag, args.dial, value, context == 'long', False, args.timeout)
                results.extend(load_tokens(tag, expected))
            if any(result != results[0] for result in results[1:]):
                raise RuntimeError(f'{context}: tokens differ across arms or ranks; timing blocked')
            print(f'{context}: both ranks and arms have identical tokens', flush=True)
        for context in ('short', 'long'):
            for value in (0, 1):
                tag = f'{session}_profile_{context}_{value}'
                run_arm(tag, args.dial, value, context == 'long', True, args.timeout)
                load_tokens(tag, {'0'} if context == 'long' else {'0', '1', '2'})
                print(f'{tag}: rank logs, rank0 trace and GPU telemetry in {LOGS}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, TimeoutError, subprocess.SubprocessError) as exc:
        print(f'ERROR: {exc}. No further arms will be launched; check for surviving ranks before retrying.', file=sys.stderr)
        sys.exit(1)
