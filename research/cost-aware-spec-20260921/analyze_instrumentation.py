#!/usr/bin/env python3
"""Summarize real timing controls and preserve their scope and raw values."""
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics
ROOT=Path(__file__).resolve().parent
RAW=ROOT/'raw'


def write(name,rows):
    with (RAW/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def main():
    lines=[s for s in (RAW/'kernel-cost.log').read_text().splitlines() if s.startswith('dimension,') or re.match(r'^\d+,\d+,\d+,',s)]
    (RAW/'kernel-cost.csv').write_text('\n'.join(lines)+'\n')
    rows=list(csv.DictReader(lines))
    if len(rows)!=2*6*8*3: raise SystemExit('kernel suite incomplete')
    summary=[]
    for d in [1024,4096]:
        for n in range(1,7):
            group=[r for r in rows if int(r['dimension'])==d and int(r['rows'])==n]
            by={(r['repeat'],r['mode']):float(r['ms_per_cycle']) for r in group}
            for mode in ['outer_timer','enqueue_timers','stage_sync_timers']:
                selected=[r for r in group if r['mode']==mode]
                vals=[float(r['ms_per_cycle']) for r in selected]
                ratio=[float(r['ms_per_cycle'])/by[r['repeat'],'outer_timer'] for r in selected]
                summary.append(dict(dimension=d,rows=n,mode=mode,median_ms=statistics.median(vals),min_ms=min(vals),max_ms=max(vals),median_paired_time_ratio=statistics.median(ratio)))
    write('kernel-cost-summary.csv',summary)
    profile=[]
    texts=[]
    for i,enabled in enumerate([False,True,True,False,True,False,False,True]):
        stem=f'streaming-profile-{i}-'+('on' if enabled else 'off')
        with (RAW/(stem+'.csv')).open() as f: values=list(csv.DictReader(f))
        if len(values)!=1 or values[0]['gen_tokens']!='128': raise SystemExit('profile suite incomplete')
        text=(RAW/(stem+'.log')).read_text()
        # Exact generated text ends before cleanup's next diagnostic line.
        match=re.search(r'ds4-bench: gen\[ctx=256\] decoded text: "(.*?)"\n(?=ds4:|$)',text,re.S)
        if not match: raise SystemExit(f'missing generated output: {stem}')
        texts.append(match[1])
        profile.append(dict(run=i,instrumentation='on' if enabled else 'off',
                            **values[0],output_sha256=hashlib.sha256(match[1].encode()).hexdigest()))
    if len(set(texts))!=1: raise SystemExit('unexplained greedy output difference')
    write('streaming-profile-all.csv',profile)
    off=[float(r['gen_tps']) for r in profile if r['instrumentation']=='off']
    on=[float(r['gen_tps']) for r in profile if r['instrumentation']=='on']
    # AB, BA, BA, AB adjacent pairs; time-on/time-off = TPS-off/TPS-on.
    paired=[]
    for i in range(0,8,2):
        pair={r['instrumentation']:float(r['gen_tps']) for r in profile[i:i+2]}
        paired.append(pair['off']/pair['on'])
    result=dict(scope='Existing SSD expert timing summary on ordinary decoding only; not new speculative-stage instrumentation.',
        off_tps=off,on_tps=on,off_median_tps=statistics.median(off),on_median_tps=statistics.median(on),
        paired_time_ratios=paired,median_paired_time_ratio=statistics.median(paired),
        greedy_output_identical=True,output_sha256=profile[0]['output_sha256'])
    (RAW/'instrumentation-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
