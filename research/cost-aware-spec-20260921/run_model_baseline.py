#!/usr/bin/env python3
"""One GPU job at a time; ordinary baseline only. Capture VM and I/O counters."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent.parent
manifest=json.loads((ROOT/'model-inputs/manifest.json').read_text())


def snapshot():
    result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    for name,cmd in [('vm_stat',['vm_stat']),('swap',['sysctl','vm.swapusage']),
                     ('memory_pressure',['memory_pressure','-Q']),
                     ('disk',['iostat','-Id']),('thermal',['pmset','-g','therm'])]:
        run=subprocess.run(cmd,capture_output=True,text=True,timeout=15)
        result[name]=dict(returncode=run.returncode,stdout=run.stdout,stderr=run.stderr)
    return result


def main():
    for item in manifest['files'].values():
        assert hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()==item['sha256']
    jobs=[('warmup','code',32,256)]
    for repeat in range(manifest['repeats']):
        # Reverse domain order on alternate rounds to expose drift.
        domains=['code','prose','structured'] if repeat%2==0 else ['structured','prose','code']
        jobs.extend((f'repeat{repeat+1}',domain,manifest['output_tokens'],2048) for domain in domains)
    for label,domain,tokens,maxctx in jobs:
        name=f'model-{label}-{domain}'
        before=snapshot()
        cmd=['python3',str(ROOT/'run_bounded.py'),'--timeout','900','--output',str(ROOT/'raw'/f'{name}.log'),
             '--','/usr/bin/time','-l','./ds4-bench','-m',manifest['model'],
             '--ssd-streaming','--ssd-streaming-cache-experts',manifest['cache'],
             '--prefill-chunk',str(manifest['prefill_chunk']),
             '--ctx-start','256','--ctx-max',str(maxctx),'--step-incr','1792',
             '--ctx-alloc',str(manifest['context_allocation']),'--gen-tokens',str(tokens),
             '--prompt-file',str(ROOT/manifest['files'][domain]['path']),
             '--csv',str(ROOT/'raw'/f'{name}.csv')]
        run=subprocess.run(cmd,cwd=REPO,timeout=930)
        after=snapshot()
        (ROOT/'raw'/f'{name}.system.json').write_text(json.dumps(dict(before=before,after=after),indent=2)+'\n')
        if run.returncode: raise SystemExit(run.returncode)
        print(name+' complete',flush=True)

if __name__=='__main__': main()
