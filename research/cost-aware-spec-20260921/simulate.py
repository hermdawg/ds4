#!/usr/bin/env python3
"""Synthetic scheduling counterfactuals. No model tokens or GPU speedups.

Each output position has a shared potential acceptance prefix. All policies
see only their executed observations. Costs are explicit assumptions from
suite.json. Actual production DSpark scheduling is called through bridge.c.
"""
import argparse
import csv
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time

ROOT = Path(__file__).resolve().parent
LIB = C.CDLL(str(ROOT/'bridge.dylib'))
for name, args, result in [
    ('research_default_new', [C.c_int], C.c_void_p),
    ('research_default_free', [C.c_void_p], None),
    ('research_default_reset', [C.c_void_p], None),
    ('research_default_skip', [C.c_void_p], C.c_int),
    ('research_default_note', [C.c_void_p, C.c_uint, C.c_int, C.c_double, C.c_double, C.c_double], None),
    ('research_default_pause', [C.c_void_p], C.c_uint),
    ('research_policy_new', [C.c_double, C.c_uint, C.c_double, C.c_int], C.c_void_p),
    ('research_policy_free', [C.c_void_p], None),
    ('research_policy_choose', [C.c_void_p, C.c_uint, C.c_uint, C.c_uint], C.c_uint),
    ('research_policy_observe', [C.c_void_p, C.c_uint, C.c_uint, C.c_double, C.c_uint], C.c_int),
]:
    f = getattr(LIB, name)
    f.argtypes, f.restype = args, result

COST_ENV = ('DS4_DSPARK_SCHEDULER_MAX_MS_PER_ACCEPT_MILLI',
            'DS4_DSPARK_SCHEDULER_MAX_EXTRA_SAVED_RATIO_MILLI',
            'DS4_DSPARK_SCHEDULER_BREAK_EVEN_WINDOW')


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:8], 'little')


def potential_cycles(case, seed, mode):
    n = case['output_tokens']
    all_rows = []
    for pos in range(n):
        rng = random.Random(stable_seed(case['id'], seed, pos))
        phase = next(p for p in case['phases'] if (pos/n) < p[0])
        _, first, later, draft_base, draft_per, verify_base, verify_per, recover = phase
        context = case['context'] + pos
        x = max(0, context-256)/2048
        target = case['target_ms'] * (1 + case['target_context_slope'] * x)
        noise = math.exp(rng.gauss(-case['noise_sigma']**2/2, case['noise_sigma']))
        no_draft = rng.random() < case['proposal_none_probability']
        hits = [rng.random() < (first if d == 0 else later) for d in range(5)]
        prefix = 0
        for hit in hits:
            if not hit:
                break
            prefix += 1
        rows = []
        for requested in range(6):
            depth = min(requested, n-pos-1)
            accepted = 0 if no_draft else min(depth, prefix)
            draft = case['target_ms'] * (draft_base + depth*draft_per) if depth else 0
            batched = mode == 'resident_seed_batch' and depth >= 3 and not no_draft
            target_stage = 0 if batched else target
            verify = 0
            if depth and not no_draft and (batched or accepted > 0):
                rows_verified = depth + int(batched)
                verify = case['target_ms'] * (verify_base + verify_per * rows_verified) * (1 + case['verify_context_slope']*x)
            recovery = case['target_ms'] * recover if verify and accepted < depth else 0
            stages = [target_stage, draft, verify, recovery]
            stages = [v*noise for v in stages]
            rows.append(dict(depth=depth, committed=1+accepted, accepted=accepted,
                no_draft=int(no_draft and depth > 0), confidence=.4 if no_draft else .8,
                target_ms=stages[0], draft_ms=stages[1], verify_ms=stages[2],
                recovery_ms=stages[3], total_ms=sum(stages),
                reference_target_ms=target*noise, context=context))
        all_rows.append(rows)
    return all_rows


def run_policy(case, seed, mode, policy, config, potential=None, trace=False):
    potential = potential if potential is not None else potential_cycles(case, seed, mode)
    for key in COST_ENV:
        os.environ.pop(key, None)
    os.environ['DS4_DSPARK_SCHEDULER'] = '1'
    if policy == 'existing_cost_gate':
        os.environ[COST_ENV[1]] = '1000'
        os.environ[COST_ENV[2]] = '4'
    default = LIB.research_default_new(mode == 'resident_seed_batch') if policy in ('default', 'existing_cost_gate') else None
    candidate = LIB.research_policy_new(config['decay'], config['probe_tokens'], config['margin'], config['context_buckets']) if policy == 'candidate' else None
    pos, cycles, draft_cycles, rejects, proposed, accepted = 0, 0, 0, 0, 0, 0
    last_target = 0.0
    totals = dict(target_ms=0., draft_ms=0., verify_ms=0., recovery_ms=0., total_ms=0.)
    depths = [0]*6
    traces = []
    begin = time.perf_counter()
    try:
        while pos < case['output_tokens']:
            remaining = case['output_tokens'] - pos
            skipped = False
            if policy == 'ordinary':
                d = 0
            elif policy.startswith('fixed_'):
                d = min(int(policy[6:]), remaining-1)
            elif default:
                if remaining < 10:
                    d = 0
                else:
                    skipped = bool(LIB.research_default_skip(default))
                    d = 0 if skipped else 5
            elif candidate:
                d = LIB.research_policy_choose(candidate, case['context']+pos, remaining, 5)
            else:
                raise ValueError(policy)
            obs = potential[pos][d]
            if obs['target_ms'] > 0:
                last_target = obs['reference_target_ms']
            if default and remaining >= 10:
                # The production cost gate compares proposal+verification+
                # recovery with accepted*the last separately evaluated target.
                extra = obs['draft_ms'] + obs['verify_ms'] + obs['recovery_ms']
                LIB.research_default_note(default, obs['accepted'], int(skipped or obs['no_draft']), extra, last_target, obs['confidence'])
            if candidate:
                assert LIB.research_policy_observe(candidate, obs['context'], d, obs['total_ms'], obs['committed'])
            for key in totals:
                totals[key] += obs[key]
            depths[d] += 1
            cycles += 1
            proposed += obs['depth'] if not obs['no_draft'] else 0
            accepted += obs['accepted']
            draft_cycles += d > 0
            rejects += d > 0 and obs['accepted'] < obs['depth']
            if trace:
                traces.append(dict(case=case['id'], seed=seed, mode=mode, policy=policy,
                    output_position=pos, requested_depth=d, **obs))
            pos += obs['committed']
        assert pos == case['output_tokens']
    finally:
        if default:
            LIB.research_default_free(default)
        if candidate:
            LIB.research_policy_free(candidate)
    result = dict(case=case['id'], split=case['split'], domain=case['domain'],
        initial_context=case['context'], seed=seed, mode=mode, policy=policy,
        committed=pos, cycles=cycles, draft_cycles=draft_cycles, rejections=rejects,
        proposed=proposed, accepted=accepted, **totals,
        ms_per_token=totals['total_ms']/pos, simulated_tps=1000*pos/totals['total_ms'],
        host_simulator_seconds=time.perf_counter()-begin)
    result.update({f'depth_{d}_cycles': count for d, count in enumerate(depths)})
    return result, traces


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare(rows):
    keyed = {(r['case'],r['seed'],r['mode'],r['policy']):r for r in rows}
    summaries=[]
    for mode in sorted({r['mode'] for r in rows}):
        for policy in sorted({r['policy'] for r in rows}):
            subset = [r for r in rows if r['mode']==mode and r['policy']==policy]
            ratios=[]
            for r in subset:
                ref = keyed[r['case'],r['seed'],mode,'ordinary']
                ratios.append(ref['total_ms']/r['total_ms'])
            summaries.append(dict(mode=mode,policy=policy, runs=len(subset),
                geometric_speedup_vs_ordinary=math.exp(statistics.mean(map(math.log,ratios))),
                min_speedup=min(ratios), max_speedup=max(ratios)))
    return summaries


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', choices=['tune','heldout'], required=True)
    p.add_argument('--config', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed-count', type=int)
    p.add_argument('--trace', action='store_true')
    a=p.parse_args()
    suite=json.loads((ROOT/'suite.json').read_text())
    expected=(ROOT/'suite.sha256').read_text().split()[0]
    assert hashlib.sha256((ROOT/'suite.json').read_bytes()).hexdigest()==expected
    config=json.loads(a.config.read_text()) if a.config else dict(decay=.8,probe_tokens=32,margin=.03,context_buckets=True)
    seeds=suite['tuning_seeds' if a.split=='tune' else 'heldout_seeds']
    if a.seed_count:
        seeds=seeds[:a.seed_count]
    rows, traces=[],[]
    policies=['ordinary','default','existing_cost_gate']+[f'fixed_{d}' for d in range(1,6)]+['candidate']
    for mode in suite['simulated_execution_modes']:
        for case in [c for c in suite['cases'] if c['split']==a.split]:
            for seed in seeds:
                potential=potential_cycles(case,seed,mode)
                for policy in policies:
                    result, detail=run_policy(case,seed,mode,policy,config,potential,a.trace and seed==seeds[0])
                    rows.append(result)
                    traces.extend(detail)
    write_csv(a.output,rows)
    if traces:
        write_csv(a.output.with_suffix('.trace.csv'),traces)
    summary=compare(rows)
    a.output.with_suffix('.summary.json').write_text(json.dumps(dict(evidence_class=suite['evidence_class'],config=config,summary=summary),indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
