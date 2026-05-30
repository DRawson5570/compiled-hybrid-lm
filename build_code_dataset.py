"""build_code_dataset.py — Tokenize Python files for code steerer cartridge."""
import sys, os
from pathlib import Path
import torch
from transformers import AutoTokenizer
from concurrent.futures import ProcessPoolExecutor

def collect_python_files(root_dirs):
    files = []
    for d in root_dirs:
        for root, _, fnames in os.walk(d):
            for fn in fnames:
                if fn.endswith('.py'):
                    files.append(os.path.join(root, fn))
    return files

def tokenize_file(filepath):
    try:
        with open(filepath, 'rb') as f:
            content = f.read().decode('utf-8', errors='replace')
        tok = AutoTokenizer.from_pretrained('gpt2')
        ids = tok.encode(content)
        return ids
    except Exception:
        return []

def main():
    out_dir = Path('/home/drawson/deepseek_experiments/artifacts/code_steerer')
    out_dir.mkdir(parents=True, exist_ok=True)

    root_dirs = [
        '/home/drawson/anaconda3/lib/python3.12/',
        '/home/drawson/anaconda3/lib/python3.12/site-packages/',
    ]

    print('[1] Collecting Python files...')
    files = collect_python_files(root_dirs)
    print(f'  {len(files)} files')

    print('[2] Tokenizing...')
    tok = AutoTokenizer.from_pretrained('gpt2')
    all_ids = []
    for i, fp in enumerate(files):
        if i % 5000 == 0:
            print(f'  {i}/{len(files)}  ({len(all_ids):,} tokens)', flush=True)
        ids = tokenize_file(fp)
        all_ids.extend(ids)

    train_ids = torch.tensor(all_ids, dtype=torch.long)
    split = int(len(train_ids) * 0.95)
    val_ids = train_ids[split:split + 50000]
    train_ids = train_ids[:split]

    torch.save(train_ids, out_dir / 'train_ids.pt')
    torch.save(val_ids, out_dir / 'validation_ids.pt')
    print(f'  Train: {len(train_ids):,}  Val: {len(val_ids):,}')
    print(f'  Saved to {out_dir}')

if __name__ == '__main__':
    main()
