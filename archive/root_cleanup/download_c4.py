import os
os.environ['HF_HOME'] = '/media/drawson/SSD-PGU3/hf_cache'
os.environ['HF_HUB_CACHE'] = '/media/drawson/SSD-PGU3/hf_cache/hub'
os.environ['HF_DATASETS_CACHE'] = '/media/drawson/SSD-PGU3/hf_cache/datasets'
import time
from datasets import load_dataset
print('[C4] Downloading compressed parquet files to SSD (not streaming)...')
t0 = time.time()
ds = load_dataset('allenai/c4', 'en', split='train', trust_remote_code=True)
print(f'  Downloaded {len(ds):,} examples in {time.time()-t0:.0f}s')
print(f'  Cache size: ', flush=True)
import subprocess
subprocess.run(['du', '-sh', '/media/drawson/SSD-PGU3/hf_cache/'])
