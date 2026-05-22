"""Pre-cache C4 and The Pile to the new ext4 SSD.

Streams through the full datasets so HuggingFace caches them locally.
Can be killed and resumed — datasets cache is incremental.
"""
import os, sys, time
os.environ['HF_HOME'] = '/media/drawson/SSD-PGU3/hf_cache'

from datasets import load_dataset

for name, config, extra_kwargs in [
    ('allenai/c4', 'en', {'trust_remote_code': True}),
    ('monology/pile-uncopyrighted', None, {}),
]:
    print(f'[{name}] Starting cache...', flush=True)
    t0 = time.time()
    total_bytes = 0
    if config:
        ds = load_dataset(name, config, split='train', streaming=True, **extra_kwargs)
    else:
        ds = load_dataset(name, split='train', streaming=True, **extra_kwargs)
    for i, example in enumerate(ds):
        text = example.get('text', '')
        total_bytes += len(text)
        if i % 100000 == 0:
            elapsed = time.time() - t0
            gb = total_bytes / 1e9
            print(f'  [{name}] {i:,} examples, {gb:.1f} GB, {elapsed:.0f}s', flush=True)
    elapsed = time.time() - t0
    print(f'[{name}] DONE: {total_bytes/1e9:.1f} GB in {elapsed:.0f}s', flush=True)
