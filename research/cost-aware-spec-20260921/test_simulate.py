#!/usr/bin/env python3
import copy
import json
import math
import os
import unittest
from simulate import LIB, ROOT, COST_ENV, potential_cycles, run_policy

class Tests(unittest.TestCase):
    def setUp(self):
        for key in COST_ENV:
            os.environ.pop(key,None)
        os.environ['DS4_DSPARK_SCHEDULER']='1'
        self.case=next(c for c in json.loads((ROOT/'suite.json').read_text())['cases'] if c['split']=='tune')
        self.config=dict(decay=.8,probe_tokens=32,margin=.03,context_buckets=True)

    def test_source_default_pause_and_reset(self):
        for seed_batch,expected in [(0,2),(1,32)]:
            s=LIB.research_default_new(seed_batch)
            try:
                for _ in range(4):
                    self.assertEqual(LIB.research_default_skip(s),0)
                    LIB.research_default_note(s,0,0,100,20,.8)
                self.assertEqual(LIB.research_default_pause(s),expected)
                for _ in range(expected):
                    self.assertEqual(LIB.research_default_skip(s),1)
                    LIB.research_default_note(s,0,1,100,20,.4)
                self.assertEqual(LIB.research_default_skip(s),0)
                LIB.research_default_note(s,0,1,10,20,.4)
                self.assertEqual(LIB.research_default_pause(s),7)
                LIB.research_default_reset(s)
                self.assertEqual(LIB.research_default_skip(s),0)
                self.assertEqual(LIB.research_default_pause(s),0)
            finally: LIB.research_default_free(s)

    def test_source_existing_cost_gate(self):
        os.environ[COST_ENV[1]]='1000'
        os.environ[COST_ENV[2]]='4'
        s=LIB.research_default_new(0)
        try:
            for _ in range(4):
                self.assertEqual(LIB.research_default_skip(s),0)
                LIB.research_default_note(s,2,0,50,20,.8)
            self.assertEqual(LIB.research_default_pause(s),4)
        finally: LIB.research_default_free(s)

    def test_total_and_output_limits(self):
        for limit in [1,2,3,5,6,9,10,11,17,32]:
            case=copy.deepcopy(self.case)
            case['output_tokens']=limit
            for mode in ['resident_seed_batch','separate_seed']:
                potential=potential_cycles(case,0,mode)
                for policy in ['ordinary','default','existing_cost_gate','candidate']+[f'fixed_{d}' for d in range(1,6)]:
                    result,trace=run_policy(case,0,mode,policy,self.config,potential,True)
                    self.assertEqual(result['committed'],limit)
                    self.assertEqual(sum(o['committed'] for o in trace),limit)
                    self.assertAlmostEqual(result['total_ms'],sum(result[k] for k in ['target_ms','draft_ms','verify_ms','recovery_ms']))
                    for obs in trace:
                        self.assertGreater(obs['total_ms'],0)
                        self.assertLessEqual(obs['committed'],obs['depth']+1)
                        self.assertLessEqual(obs['output_position']+obs['committed'],limit)

    def test_analytical_cycle_costs(self):
        case=copy.deepcopy(self.case)
        case.update(noise_sigma=0, proposal_none_probability=0, target_context_slope=0,verify_context_slope=0)
        case['phases']=[[1,1,1,.1,.02,.8,.1,.3]]
        for mode in ['resident_seed_batch','separate_seed']:
            rows=potential_cycles(case,0,mode)[0]
            self.assertEqual(rows[0]['total_ms'],20)
            self.assertEqual(rows[1]['total_ms'],20+20*.12+20*.9)
            self.assertEqual(rows[5]['committed'],6)
            expected=20*.2+20*(.8+.1*(6 if mode=='resident_seed_batch' else 5))
            if mode=='separate_seed': expected+=20
            self.assertAlmostEqual(rows[5]['total_ms'],expected)
        case['phases'][0][1]=0
        rows=potential_cycles(case,0,'separate_seed')[0]
        self.assertEqual(rows[5]['committed'],1)
        self.assertEqual(rows[5]['verify_ms'],0)
        self.assertEqual(rows[5]['recovery_ms'],0)
        self.assertAlmostEqual(rows[5]['total_ms'],24)
        rows=potential_cycles(case,0,'resident_seed_batch')[0]
        self.assertEqual(rows[5]['committed'],1)
        self.assertAlmostEqual(rows[5]['total_ms'],38)

    def test_shared_first_prefix(self):
        for mode in ['resident_seed_batch','separate_seed']:
            for rows in potential_cycles(self.case,7,mode):
                for d in range(1,6):
                    self.assertGreaterEqual(rows[d]['accepted'],rows[d-1]['accepted'])
                    self.assertLessEqual(rows[d]['accepted'],rows[d-1]['accepted']+1)

if __name__=='__main__': unittest.main(verbosity=2)
