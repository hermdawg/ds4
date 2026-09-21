#!/usr/bin/env python3
"""Run one child process group with a timeout; save exact command and metadata."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--timeout', type=float, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        p.error('missing command')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    begin = time.monotonic()
    timed_out = False
    interrupted = 0
    code = 125
    with args.output.open('w') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        def stop_group():
            if child.poll() is not None:
                return
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()

        def interrupted_handler(signum, _frame):
            nonlocal interrupted
            interrupted = signum
            raise InterruptedError('runner interrupted')

        # Outer campaign timeouts must also stop this runner's child session.
        # Otherwise a nested start_new_session child could outlive its parent.
        signal.signal(signal.SIGTERM, interrupted_handler)
        signal.signal(signal.SIGINT, interrupted_handler)
        try:
            code = child.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            code = 124
            stop_group()
        except InterruptedError:
            code = 128 + interrupted
            stop_group()
    metadata = dict(command=command, cwd=os.getcwd(), started=started,
                    elapsed_seconds=time.monotonic()-begin, exit_code=code,
                    timed_out=timed_out, interrupted_signal=interrupted)
    args.output.with_suffix(args.output.suffix+'.json').write_text(
        json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata), flush=True)
    raise SystemExit(code)

if __name__ == '__main__':
    main()
