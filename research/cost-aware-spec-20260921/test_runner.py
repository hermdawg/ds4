#!/usr/bin/env python3
"""Check that both direct and nested timeouts stop child process groups."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
RUNNER=Path(__file__).resolve().with_name('run_bounded.py')

class RunnerTests(unittest.TestCase):
    def test_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            log=Path(tmp)/'ok.log'
            r=subprocess.run([sys.executable,str(RUNNER),'--timeout','2','--output',str(log),'--',sys.executable,'-c','print(42)'],capture_output=True,text=True,timeout=5)
            self.assertEqual(r.returncode,0)
            self.assertEqual(log.read_text(),'42\n')
            self.assertEqual(json.loads(log.with_suffix('.log.json').read_text())['exit_code'],0)

    def test_direct_and_nested_timeout(self):
        for nested in [False,True]:
            with self.subTest(nested=nested), tempfile.TemporaryDirectory() as tmp:
                tmp=Path(tmp)
                child_code='import os,time;print(os.getpid(),flush=True);time.sleep(30)'
                inner=[sys.executable,str(RUNNER),'--timeout','30' if nested else '.2','--output',str(tmp/'inner.log'),'--',sys.executable,'-c',child_code]
                cmd=[sys.executable,str(RUNNER),'--timeout','.5','--output',str(tmp/'outer.log'),'--',*inner] if nested else inner
                start=time.monotonic()
                r=subprocess.run(cmd,capture_output=True,text=True,timeout=10)
                self.assertEqual(r.returncode,124)
                self.assertLess(time.monotonic()-start,8)
                pid=int((tmp/'inner.log').read_text().strip())
                with self.assertRaises(ProcessLookupError): os.kill(pid,0)
                inner_meta=json.loads((tmp/'inner.log.json').read_text())
                self.assertEqual(inner_meta['exit_code'],143 if nested else 124)

if __name__=='__main__': unittest.main(verbosity=2)
