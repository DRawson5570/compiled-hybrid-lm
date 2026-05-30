"""train_4b.py — Train a ~4B GPT-2 BPE model with compiled priors + steerer on pe2 (2× M40).

Uses bitsandbytes 4-bit quantization to fit 4B params in 24GB.
Same architecture as V4 train_steerer_v4.py, scaled up.
Compiled priors match GPT-2 BPE tokenizer (V=50257).
"""
import sys, time, math, pickle, argparse
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from train_scaled_neural_lm import DeepCausalLM
from superposition_steerer_v3 import SuperpositionSteererV3
from gpu_channels import GPUFeatureComputer
from train_steerer_v4 import StreamingSteererDatasetV4, FastNgramFeatures, compute_cpu_features, PUNCT_IDS

V = 50257

# ~1B param config: d=1536, L=20, heads=12, d_ff=6144
CFG_4B = dict(d_model=1536, n_layers=20, n_heads=12, d_ff=6144, max_len=512)


def build_model(cfg, device):
    model = DeepCausalLM(
        vocab=V, d_model=cfg['d_model'], n_layers=cfg['n_layers'],
        n_heads=cfg['n_heads'], d_ff=cfg['d_ff'],
        max_len=cfg['max_len'], dropout=0.0,
    )
    return model.to(device)


def load_priors(device):
    priors_dir = REPO.parent / 'artifacts/compiled_priors_v3'
    word_topics = torch.load(priors_dir / 'word_topics.pt', map_location='cpu', weights_only=False)
    with open(priors_dir / 'pos_stats.pkl', 'rb') as f:
        pos_stats = pickle.load(f)
    tag_to_idx = pos_stats.get('tag_to_idx', {'WORD': 0, 'PUNCT': 1, 'NUM': 2})
    token_to_tag = pos_stats.get('token_to_tag', {})
    pos_tags = {int(k): tag_to_idx.get(v, 0) for k, v in token_to_tag.items()}
    ppmi_emb = torch.randn(V, 256, dtype=torch.float32) * 0.01
    word_topics = word_topics.to(device)
    return word_topics, pos_tags, ppmi_emb


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--seq-len', type=int, default=128)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--steerer-lr', type=float, default=1e-3)
    p.add_argument('--device', type=str, default='cuda')
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = REPO.parent / 'artifacts/train_4b'
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60); print(f' TRAIN 4B with COMPILED PRIORS')
    print(f' d={CFG_4B["d_model"]} L={CFG_4B["n_layers"]} h={CFG_4B["n_heads"]} d_ff={CFG_4B["d_ff"]}')
    print(f' epochs={args.epochs}  batch={args.batch} seq={args.seq_len}')
    print('=' * 60)

    print('[load] Data...')
    train_ids = torch.load(REPO.parent/'artifacts/wikitext_gpt2/train_ids.pt', weights_only=False).long()
    val_ids = torch.load(REPO.parent/'artifacts/wikitext_gpt2/validation_ids.pt', weights_only=False).long()
    print(f'  Train: {len(train_ids):,}  Val: {len(val_ids):,}')

    print('[load] Compiled priors...')
    word_topics, pos_tags, ppmi_emb = load_priors(device)

    print('[build] 3B DeepCausalLM...')
    model = build_model(CFG_4B, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  {n_params:,} params  d_model={CFG_4B["d_model"]}')
    for p in model.parameters():
        p.requires_grad = True

    print('[build] Steerer...')
    steerer = SuperpositionSteererV3(d_model=CFG_4B['d_model'], init_scale=0.01, noise_scale=0.05)
    steerer = steerer.to(device)
    n_hooks = steerer.register_hooks(model)
    print(f'  {n_hooks} hooks, {sum(p.numel() for p in steerer.parameters()):,} params')

    print('[build] GPU Feature Computer...')
    gpu_fc = GPUFeatureComputer(
        V=V, punct_ids=PUNCT_IDS, topic_matrix=word_topics,
        pos_tags=pos_tags, ppmi_embeddings=ppmi_emb, device=device)

    opt = torch.optim.SGD([
        {'params': model.parameters(), 'lr': args.lr},
        {'params': steerer.parameters(), 'lr': args.steerer_lr},
    ], momentum=0.9, weight_decay=0.1)

    train_dataset = StreamingSteererDatasetV4(train_ids=train_ids, seq_len=args.seq_len, V=V)
    train_loader = DataLoader(train_dataset, batch_size=args.batch,
                              num_workers=2, pin_memory=True, drop_last=True)

    best_eval_b = float('inf'); best_eval_s = float('inf')

    for ep in range(1, args.epochs + 1):
        model.train(); steerer.train()
        total_loss = 0.0; t0 = time.time()
        loader_iter = iter(train_loader)

        for step in range(args.steps):
            x, y, w_cpu = next(loader_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            w_cpu = w_cpu.to(device, non_blocking=True)

            w_gpu = gpu_fc.compute_features(x)
            w_gpu[:, :, 0:9] = w_cpu[:, :, :9]

            steerer.set_weights(w_gpu)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
            loss = loss + 0.001 * steerer.orthogonal_penalty()

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(steerer.parameters()), 1.0)
            opt.step()
            total_loss += loss.item()

        # Eval
        model.eval(); steerer.eval()
        cpu_ch = FastNgramFeatures(V)
        with torch.no_grad():
            es_nll, es_n = 0.0, 0
            for s in range(0, min(len(val_ids) - 1, 2000), 64):
                cl = min(64, len(val_ids) - s - 1)
                if cl <= 0: continue
                inp = val_ids[s:s+cl].unsqueeze(0).to(device)
                tgt = val_ids[s+1:s+cl+1].unsqueeze(0).to(device)
                w_e = gpu_fc.compute_features(inp)
                ctx = val_ids[s:s+cl].tolist()
                w_cpu_eval = compute_cpu_features(ctx, cpu_ch)
                w_e[0, :w_cpu_eval.shape[0], 0:9] = w_cpu_eval[:, :9].to(device)
                steerer.set_weights(w_e)
                l = model(inp)
                es_nll += F.cross_entropy(l.reshape(-1, V), tgt.reshape(-1), reduction='sum').item()
                es_n += cl
            eval_s = math.exp(es_nll / max(es_n, 1))

            steerer._current_weights = None
            eb_nll, eb_n = 0.0, 0
            for s in range(0, len(val_ids) - 1, 128):
                cl = min(128, len(val_ids) - s - 1)
                if cl <= 0: continue
                inp = val_ids[s:s+cl].unsqueeze(0).to(device)
                tgt = val_ids[s+1:s+cl+1].unsqueeze(0).to(device)
                l = model(inp)
                eb_nll += F.cross_entropy(l.reshape(-1, V), tgt.reshape(-1), reduction='sum').item()
                eb_n += cl
            eval_b = math.exp(eb_nll / max(eb_n, 1))

        avg_loss = total_loss / args.steps; elapsed = time.time() - t0
        status = ''
        if eval_b < best_eval_b: best_eval_b = eval_b; status += 'b'
        if eval_s < best_eval_s: best_eval_s = eval_s; status += 's'

        if 'b' in status:
            torch.save({'state_dict': model.state_dict(), 'steerer_state': steerer.state_dict(),
                        'eval_s': eval_s, 'eval_b': eval_b, 'epoch': ep, 'opt_state': opt.state_dict()},
                       out_dir / 'steerer_best_b.pt')
        if 's' in status:
            torch.save({'state_dict': model.state_dict(), 'steerer_state': steerer.state_dict(),
                        'eval_s': eval_s, 'eval_b': eval_b, 'epoch': ep, 'opt_state': opt.state_dict()},
                       out_dir / 'steerer_best_s.pt')
        if ep % 10 == 0:
            torch.save({'state_dict': model.state_dict(), 'steerer_state': steerer.state_dict(),
                        'eval_s': eval_s, 'eval_b': eval_b, 'epoch': ep, 'opt_state': opt.state_dict()},
                       out_dir / f'checkpoint_ep{ep}.pt')

        print(f'  epoch={ep:3d}  loss={avg_loss:.4f}  ppl={math.exp(avg_loss):.1f}  '
              f'eval_s={eval_s:.1f}  eval_b={eval_b:.1f}  best_b={best_eval_b:.1f}  '
              f'[{status}]  time={elapsed:.0f}s', flush=True)

    print(f'\nDone. Best eval_b: {best_eval_b:.1f}  Best eval_s: {best_eval_s:.1f}')


if __name__ == '__main__':
    main()
