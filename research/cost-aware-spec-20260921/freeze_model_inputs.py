#!/usr/bin/env python3
"""Freeze raw continuation corpora for the ordinary-only local baseline."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
out=ROOT/'model-inputs'
out.mkdir(exist_ok=True)
corpora={
'code': '\n'.join(f'def test_cache_entry_{i}(cache):\n    cache.put("key_{i}", {i})\n    assert cache.get("key_{i}") == {i}\n' for i in range(320)),
'prose': '\n'.join(f'The cartographer marked inlet {i} on the map. At low tide, a narrow path appeared between the wet rocks. A fishing boat waited beyond the headland, and the village lamps faded as the sun rose.' for i in range(200)),
'structured': '\n'.join(json.dumps(dict(tool='create_event' if i%2==0 else 'update_event',arguments=dict(event_id=f'event_{i}',title=f'Meeting {i}',duration_minutes=30,calendar='work'))) for i in range(300)),
}
files={}
for domain,text in corpora.items():
    path=out/f'{domain}.txt'
    if path.exists(): raise SystemExit('refusing to overwrite frozen corpus')
    path.write_text(text+'\n')
    files[domain]=dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
manifest=dict(evidence_class='REAL MODEL, ORDINARY DECODE ONLY, SSD STREAMING',
    model='/Users/herman/code/open-source/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf',
    model_bytes=86720111488,source_commit='9139e2a',contexts=[256,2048],output_tokens=256,
    cache='8GB',context_allocation=4096,prefill_chunk=256,temperature=0,
    eos='benchmark selects non-EOS argmax',warmups=1,repeats=3,
    note='Raw continuation inputs sliced at exact token frontiers by unmodified ds4-bench. These are independent of the chat prompt suite and synthetic assumed costs. No speculative/model speedup comparison is possible.',files=files)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
