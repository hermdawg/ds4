#!/usr/bin/env python3
"""Tune on preregistered tuning seeds only. Holdout stays unread by this loop."""
import argparse
import itertools
import json
import math
import statistics
from simulate import ROOT, potential_cycles, run_policy, write_csv
parser=argparse.ArgumentParser()
parser.add_argument('--h2',action='store_true')
args=parser.parse_args()
suite=json.loads((ROOT/'suite.json').read_text())
cases=[c for c in suite['cases'] if c['split']=='tune']
cache=[]
for mode in suite['simulated_execution_modes']:
    for case in cases:
        for seed in suite['tuning_seeds']:
            pot=potential_cycles(case,seed,mode)
            ordinary,_=run_policy(case,seed,mode,'ordinary',{},pot)
            default,_=run_policy(case,seed,mode,'default',{},pot)
            cache.append((case,seed,mode,pot,ordinary['total_ms'],default['total_ms']))
rows=[]
grid=itertools.product([.5,.8,.95],[16,32,64],[63,41,33]) if args.h2 else itertools.product([0.0,.5,.8,.95],[16,32,64,128],[63])
for decay,probe,mask in grid:
    config=dict(decay=decay,probe_tokens=probe,margin=.03,context_buckets=True,action_mask=mask)
    ratios,vs_default,groups=[],[],{}
    for case,seed,mode,pot,ordinary,default in cache:
        result,_=run_policy(case,seed,mode,'candidate',config,pot)
        ratio=result['total_ms']/ordinary
        ratios.append(ratio)
        vs_default.append(result['total_ms']/default)
        groups.setdefault((case['id'],mode),[]).append(ratio)
    # Predeclared conservative selection: minimize mean normalized elapsed
    # time, with an additional penalty for scenario means >5% slower than
    # ordinary. No oracle or heldout information is exposed to the policy.
    worst=max(map(statistics.mean,groups.values()))
    score=statistics.mean(ratios)+max(0,worst-1.05)
    row=dict(**config,mean_time_ratio=statistics.mean(ratios),worst_case_mean_ratio=worst,
        geomean_speedup_vs_default=math.exp(-statistics.mean(map(math.log,vs_default))),
        selection_score=score)
    rows.append(row)
    print(json.dumps(row),flush=True)
suffix='-h2' if args.h2 else ''
write_csv(ROOT/f'raw/tuning-grid{suffix}.csv',rows)
best=min(rows,key=lambda r:r['selection_score'])
config={k:best[k] for k in ['decay','probe_tokens','margin','context_buckets','action_mask']}
(ROOT/f'candidate-config{suffix}.json').write_text(json.dumps(config,indent=2)+'\n')
print('SELECTED '+json.dumps(config),flush=True)
