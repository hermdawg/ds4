#!/usr/bin/env python3
"""Frozen sensitivity matrix. All outcomes here are SYNTHETIC assumptions."""
import copy
import json
from simulate import ROOT, potential_cycles, run_policy, write_csv, compare
suite=json.loads((ROOT/'suite.json').read_text())
configs={name:json.loads((ROOT/f'candidate-config{suffix}.json').read_text()) for name,suffix in [('h1',''),('h2','-h2')]}
policies=['ordinary','default','existing_cost_gate']+[f'fixed_{d}' for d in range(1,6)]+['h1','h2']
base=[c for c in suite['cases'] if c['split']=='heldout']


def cases_for(kind):
    cases=copy.deepcopy(base)
    for case in cases:
        case['id']=kind+'-'+case['id']
        if kind=='short': case['output_tokens']=32
        elif kind=='long': case['output_tokens']=2048
        elif kind=='context_boundary': case['context']=960
        elif kind=='noisy': case['noise_sigma']=.30
        elif kind in ('stable_easy','stable_hard'):
            phase=case['phases'][0].copy()
            phase[0]=1
            phase[1]=phase[2]=.95 if kind=='stable_easy' else .10
            case['phases']=[phase]
        elif kind in ('abrupt_recovery','abrupt_collapse'):
            first,second=case['phases'][0].copy(),case['phases'][0].copy()
            first[0],second[0]=.5,1
            first[1]=first[2]=.05 if kind=='abrupt_recovery' else .98
            second[1]=second[2]=.98 if kind=='abrupt_recovery' else .05
            case['phases']=[first,second]
    return cases


def adjust(potential,kind):
    for rows in potential:
        if kind=='flat_draft':
            full=rows[5]['draft_ms']
            for row in rows[1:]:
                if row['depth']:
                    row['total_ms']+=full-row['draft_ms']
                    row['draft_ms']=full
        elif kind in ('expensive_verify','cheap_verify'):
            factor=2 if kind=='expensive_verify' else .5
            for row in rows:
                row['total_ms']+=(factor-1)*row['verify_ms']
                row['verify_ms']*=factor


kinds=['fresh_seeds','flat_draft','expensive_verify','cheap_verify','short','long','context_boundary','noisy','stable_easy','stable_hard','abrupt_recovery','abrupt_collapse']
all_summary=[]
for kind in kinds:
    rows=[]
    seeds=range(400,440) if kind=='fresh_seeds' else suite['stress_seeds']
    for mode in suite['simulated_execution_modes']:
        for case in cases_for(kind):
            for seed in seeds:
                potential=potential_cycles(case,seed,mode)
                adjust(potential,kind)
                for policy in policies:
                    name='candidate' if policy in configs else policy
                    config=configs.get(policy,{})
                    result,_=run_policy(case,seed,mode,name,config,potential)
                    result['policy']=policy
                    result['stress']=kind
                    rows.append(result)
    write_csv(ROOT/f'raw/stress-{kind}.csv',rows)
    summary=compare(rows)
    all_summary.extend(dict(stress=kind,**s) for s in summary)
    print(json.dumps(dict(stress=kind,summary=[s for s in summary if s['policy'] in ('h1','h2','default')])),flush=True)
write_csv(ROOT/'raw/stress-summary.csv',all_summary)
