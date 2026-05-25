import os
os.environ['HF_HOME'] = '/media/drawson/SSD-PGU3/hf_cache'
os.environ['HF_HUB_CACHE'] = '/media/drawson/SSD-PGU3/hf_cache/hub'
os.environ['HF_DATASETS_CACHE'] = '/media/drawson/SSD-PGU3/hf_cache/datasets'
import time
from datasets import load_dataset
print('[C4] Starting cache...', flush=True)
t0 = time.time()
total_bytes = 0
ds = load_dataset('allenai/c4', 'en', split='train', streaming=True, trust_remote_code=True)
for i, example in enumerate(ds):
    total_bytes += len(example['text'])
    if i % 100000 == 0:
        gb = total_bytes / 1e9
        print(f'  {i:,} examples, {gb:.1f} GB, {time.time()-t0:.0f}s', flush=True)
print(f'[C4] DONE: {total_bytes/1e9:.1f} GB in {time.time()-t0:.0f}s', flush=True)
