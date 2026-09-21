#!/usr/bin/env python3
"""Reproducible paired summaries; simulated ratios remain labeled synthetic."""
import csv
import json
import math
from pathlib import Path
import random
import re
import statistics

ROOT=Path(__file__).resolve().parent
RAW=ROOT/'raw'


def read(name):
    with (RAW/name).open() as f: return list(csv.DictReader(f))


def gm(xs): return math.exp(statistics.mean(map(math.log,xs)))


def bootstrap_seed_ci(rows,policy,reference):
    key={(r['case'],r['seed'],r['policy']):float(r['total_ms']) for r in rows}
    seeds=sorted({r['seed'] for r in rows})
    by_seed=[]
    for seed in seeds:
        ratios=[key[r['case'],seed,reference]/float(r['total_ms']) for r in rows if r['seed']==seed and r['policy']==policy]
        by_seed.append(statistics.mean(map(math.log,ratios)))
    rng=random.Random(719)
    values=sorted(math.exp(statistics.mean(rng.choices(by_seed,k=len(by_seed)))) for _ in range(5000))
    return values[125],values[4874]


def summarize(rows):
    output=[]
    for mode in sorted({r['mode'] for r in rows}):
        m=[r for r in rows if r['mode']==mode]
        key={(r['case'],r['seed'],r['policy']):float(r['total_ms']) for r in m}
        for policy in sorted({r['policy'] for r in m}):
            selected=[r for r in m if r['policy']==policy]
            ratios=[key[r['case'],r['seed'],'ordinary']/float(r['total_ms']) for r in selected]
            current=[key[r['case'],r['seed'],'default']/float(r['total_ms']) for r in selected]
            ci=bootstrap_seed_ci(m,policy,'default')
            output.append(dict(mode=mode,policy=policy,runs=len(selected),
                speedup_vs_ordinary=gm(ratios),speedup_vs_default=gm(current),
                vs_default_ci_low=ci[0],vs_default_ci_high=ci[1],
                runs_slower_than_ordinary=sum(r<1 for r in ratios),
                runs_slower_than_default=sum(r<1 for r in current),
                worst_vs_ordinary=min(ratios),worst_vs_default=min(current)))
    return output


def csvout(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    for repeat in range(1,4):
        for domain in ('code','prose','structured'):
            path=RAW/f'model-repeat{repeat}-{domain}.csv'
            if not path.exists():
                raise SystemExit(f'Model suite incomplete: missing {path.name}')
            completed=read(path.name)
            if len(completed)!=2 or any(int(r['gen_tokens'])!=256 for r in completed):
                raise SystemExit(f'Model suite incomplete: {path.name}')
    allsummaries={}
    for stem in ['heldout-h1','heldout-h2-exploratory','stress-fresh_seeds']:
        rows=read(stem+'.csv')
        summary=summarize(rows)
        allsummaries[stem]=summary
        csvout(RAW/(stem+'.paired.csv'),summary)
        per_case=[]
        for case in sorted({r['case'] for r in rows}):
            for s in summarize([r for r in rows if r['case']==case]):
                per_case.append(dict(case=case,**s))
        csvout(RAW/(stem+'.by-case.csv'),per_case)
    model=[]
    for path in sorted(RAW.glob('model-repeat[123]-*.csv')):
        match=re.match(r'model-repeat(\d+)-(.*)',path.stem)
        for row in read(path.name):
            model.append(dict(repeat=int(match[1]),domain=match[2],**row))
    csvout(RAW/'model-all.csv',model)
    model_summary=[]
    for domain in ('code','prose','structured'):
        for context in (256,2048):
            selected=[r for r in model if r['domain']==domain and int(r['ctx_tokens'])==context]
            if not selected: continue
            vals=[float(r['gen_tps']) for r in selected]
            model_summary.append(dict(domain=domain,context=context,runs=len(vals),
                median_tps=statistics.median(vals),min_tps=min(vals),max_tps=max(vals),
                mean_tps=statistics.mean(vals),stdev_tps=statistics.stdev(vals) if len(vals)>1 else 0,
                tokens_per_run=int(selected[0]['gen_tokens']),
                median_prefill_tps=statistics.median(float(r['prefill_tps']) for r in selected)))
    csvout(RAW/'model-summary.csv',model_summary)
    (RAW/'analysis.json').write_text(json.dumps(dict(evidence='Paired synthetic comparisons; confidence intervals describe only stochastic seeds in the invented regimes, not real-prompt generalization.',synthetic=allsummaries,ordinary_streamed_model=model_summary),indent=2)+'\n')
    print(json.dumps(dict(candidate_comparisons={k:[s for s in v if s['policy'] in ('candidate','h1','h2')] for k,v in allsummaries.items()},model=model_summary),indent=2))

if __name__=='__main__': main()
