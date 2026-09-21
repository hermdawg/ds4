#!/usr/bin/env python3
"""Materialize the preregistered suite. Do not run to retune held-out inputs."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
prompts = {
    'tune': {
        'code': 'Write a complete C string-key hash table with insert, find, delete and a test main. Output only C code.',
        'prose': 'Describe a rainy railway station from the perspective of a lost umbrella. Use varied concrete details and no repeated phrases.',
        'structured': 'Produce a JSON array of 20 fictional inventory items. Each object has sku, name, quantity and reorder_level. Output only JSON.',
    },
    'heldout': {
        'code': 'Implement a Python LRU cache using a doubly linked list and a dictionary. Include get, put and unit tests. Output only code.',
        'prose': 'Write a scene in which a cartographer realizes the coast has changed overnight. Avoid repeated images and explain nothing outside the scene.',
        'structured': 'Return 20 newline-delimited JSON tool calls for a fictional calendar. Alternate create_event and update_event, with realistic arguments. Output only JSON lines.',
    },
}
# These are invented cost/acceptance regimes, not estimates for the prompts.
# Costs are relative to a 20 ms target decode at 256 tokens. Source QA reports
# are NOT converted into cycle traces. A phase is (fraction, p_first, p_later,
# draft_base, draft_per_depth, verify_base, verify_per_row, recovery).
regimes = {
    'tune': {
        'code': [[1, .90, .84, .12, .045, .82, .085, .10]],
        'prose': [[1, .30, .42, .22, .06, .95, .14, .16]],
        'structured': [[.45, .40, .50, .16, .05, .86, .09, .12], [1, .96, .93, .16, .05, .86, .09, .12]],
    },
    'heldout': {
        'code': [[.60, .86, .80, .15, .05, .85, .10, .14], [1, .55, .58, .15, .05, .85, .10, .14]],
        'prose': [[.55, .26, .38, .25, .065, 1.0, .15, .18], [1, .68, .64, .25, .065, 1.0, .15, .18]],
        'structured': [[.30, .94, .89, .19, .04, .88, .11, .13], [.60, .22, .30, .19, .04, .88, .11, .13], [1, .92, .90, .19, .04, .88, .11, .13]],
    },
}
cases = []
for split in ('tune', 'heldout'):
    for domain in ('code', 'prose', 'structured'):
        for context in (256, 2048):
            cases.append(dict(id=f'{split}-{domain}-{context}', split=split,
                domain=domain, context=context, output_tokens=256,
                prompt=prompts[split][domain], phases=regimes[split][domain],
                target_ms=20.0, target_context_slope=.30,
                verify_context_slope=.70, noise_sigma=.08,
                proposal_none_probability=.08 if domain == 'prose' else .02))
suite = dict(version=1, frozen_utc='2026-09-21T20:50:00Z',
    evidence_class='SYNTHETIC: all costs and acceptance probabilities are assumptions',
    model_baseline='DeepSeek V4 Flash 0731 IQ2XXS/w2Q2K; speculative weights unavailable',
    sampling=dict(temperature=0, top_p=1, min_p=0, seed=12345),
    fixed_depths=[0, 1, 2, 3, 4, 5],
    simulated_execution_modes=['resident_seed_batch', 'separate_seed'],
    tuning_seeds=list(range(20)), heldout_seeds=list(range(100, 140)),
    stress_seeds=list(range(200, 240)), cases=cases,
    validation_gate='Do not integrate a policy into model inference without model-level correctness and interleaved speed validation.')
text = json.dumps(suite, indent=2)+'\n'
path = ROOT/'suite.json'
if path.exists():
    raise SystemExit('Suite already frozen; refusing overwrite')
path.write_text(text)
(ROOT/'suite.sha256').write_text(hashlib.sha256(text.encode()).hexdigest()+'  suite.json\n')
