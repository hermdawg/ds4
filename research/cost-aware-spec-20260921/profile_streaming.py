#!/usr/bin/env python3
"""Serial ABBA/BAAB ordinary-decode instrumentation control, not speculation."""
import json
import os
from pathlib import Path
import subprocess
from run_model_baseline import snapshot
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent.parent
m=json.loads((ROOT/'model-inputs/manifest.json').read_text())
for run,enabled in enumerate([False,True,True,False,True,False,False,True]):
    label=f'streaming-profile-{run}-'+('on' if enabled else 'off')
    env=os.environ.copy()
    env.pop('DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY',None)
    env.pop('DS4_METAL_STREAMING_EXPERT_PROFILE_SUMMARY',None)
    if enabled: env['DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY']='1'
    cmd=['python3',str(ROOT/'run_bounded.py'),'--timeout','300','--output',str(ROOT/'raw'/f'{label}.log'),'--',
         './ds4-bench','-m',m['model'],'--ssd-streaming','--ssd-streaming-cache-experts',m['cache'],
         '--prefill-chunk',str(m['prefill_chunk']),'--ctx-start','256','--ctx-max','256',
         '--ctx-alloc',str(m['context_allocation']),'--gen-tokens','128','--show-output',
         '--prompt-file',str(ROOT/m['files']['code']['path']),
         '--csv',str(ROOT/'raw'/f'{label}.csv')]
    before=snapshot()
    r=subprocess.run(cmd,cwd=REPO,env=env,timeout=330)
    after=snapshot()
    (ROOT/'raw'/f'{label}.system.json').write_text(json.dumps(dict(instrumentation=enabled,before=before,after=after),indent=2)+'\n')
    if r.returncode: raise SystemExit(r.returncode)
    print(label+' complete',flush=True)
