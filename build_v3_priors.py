"""build_v3_priors.py — Precompute V3 compiled priors from training data.

Builds:
  1. Word-topic matrix from sparse PPMI co-occurrence (no SVD needed)
  2. POS transition matrix from NLTK tagging

Saves to artifacts/compiled_priors_v3/
"""
import sys, time, math, pickle
from pathlib import Path
import numpy as np
import torch
from collections import defaultdict, Counter

DEEPSEEK = Path('/home/drawson/deepseek_experiments')
sys.path.insert(0, str(DEEPSEEK))

# Inline build_ppmi_stats (was in archived train_steerer_v2.py)
def build_ppmi_stats(train_ids, V, max_tokens=500000):
    use_tokens = train_ids[:min(len(train_ids), max_tokens)].long().numpy()
    T = len(use_tokens)
    pair_counts = {}
    unigram_counts = np.zeros(V, dtype=np.float64)
    for t in range(1, T):
        ctx = int(use_tokens[t - 1]); tgt = int(use_tokens[t])
        unigram_counts[ctx] += 1; unigram_counts[tgt] += 1
        pair_counts[(ctx, tgt)] = pair_counts.get((ctx, tgt), 0) + 1
    total_bigrams = T - 1
    ppmi = {}
    for (ctx, tgt), cnt in pair_counts.items():
        p_ctx = unigram_counts[ctx] / total_bigrams
        p_tgt = unigram_counts[tgt] / total_bigrams
        p_joint = cnt / total_bigrams
        ppmi_val = max(0.0, np.log2(p_joint / (p_ctx * p_tgt + 1e-12)))
        if ppmi_val > 0:
            ppmi[(int(ctx), int(tgt))] = float(ppmi_val)
    return {'ppmi': ppmi, 'ppmi_norm': {}, 'total_bigrams': total_bigrams}


def build_topic_from_sparse(ppmi_stats, V, K=50, top_n=200):
    """Build word-topic matrix from sparse PPMI co-occurrence.
    
    For each token, build a topic signature from its top-N co-occurring tokens.
    Cluster signatures with mini-batch k-means to get K topics.
    """
    from sklearn.cluster import MiniBatchKMeans

    ppmi = ppmi_stats['ppmi']  # dict (ctx, tgt) -> PPMI
    
    # Build co-occurrence counts for each token
    print(f'  Building token co-occurrence vectors...')
    cooccur = defaultdict(Counter)
    for (ctx, tgt), val in ppmi.items():
        if ctx < V and tgt < V and val > 0:
            cooccur[ctx][tgt] += val
    
    # Build dense signatures for most frequent tokens
    print(f'  Building topic signatures...')
    freq_tokens = sorted(cooccur.keys(), key=lambda t: sum(cooccur[t].values()), reverse=True)
    n_tokens = min(len(freq_tokens), 50000)
    freq_tokens = set(freq_tokens[:n_tokens])
    
    import random
    sample_tokens = random.sample(list(freq_tokens), min(10000, len(freq_tokens)))
    
    signatures = np.zeros((len(sample_tokens), V), dtype=np.float32)
    token_map = {}
    for i, tok in enumerate(sample_tokens):
        if i % 2000 == 0:
            print(f'    {i}/{len(sample_tokens)}', flush=True)
        vec = np.zeros(V, dtype=np.float32)
        total = sum(cooccur[tok].values()) + 0.1
        for tgt, cnt in cooccur[tok].most_common(top_n):
            vec[tgt] = cnt / total
        signatures[i] = vec
        token_map[i] = tok
    
    print(f'  Clustering {len(sample_tokens)} signatures into {K} topics...')
    km = MiniBatchKMeans(n_clusters=K, random_state=42, batch_size=512,
                         n_init=3, max_iter=50)
    km.fit(signatures)
    centers = km.cluster_centers_.astype(np.float32)  # (K, V)
    
    # Assign all tokens to topics via top co-occurring overlap
    print(f'  Assigning all {V} tokens to topics...')
    word_topics = np.zeros((V, K), dtype=np.float32)
    assigned = 0
    for tok in range(V):
        if tok in cooccur:
            vec = np.zeros(V, dtype=np.float32)
            total = sum(cooccur[tok].values()) + 0.1
            for tgt, cnt in cooccur[tok].most_common(top_n):
                vec[tgt] = cnt / total
            # Dot product with each center
            for k in range(K):
                word_topics[tok, k] = np.dot(vec, centers[k])
            assigned += 1
    
    # Softmax normalize
    for tok in range(V):
        row = word_topics[tok]
        mx = row.max()
        if mx > 0:
            row = np.exp((row - mx) * 3.0)
            word_topics[tok] = row / (row.sum() + 1e-8)
    
    print(f'  Assigned {assigned}/{V} tokens to topics')
    return torch.tensor(word_topics)


def build_pos_transitions(train_ids, V, max_tokens=2000000):
    """Build POS transition matrix using NLTK perceptron tagger.
    
    Tags first max_tokens of corpus, compiles P(POS_t | POS_{t-1}, POS_{t-2}).
    """
    import nltk
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('universal_tagset', quiet=True)
    
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('gpt2')
    
    # Decode tokens to text for POS tagging
    print(f'  Decoding {max_tokens} tokens for POS tagging...')
    use_tokens = train_ids[:min(len(train_ids), max_tokens)].numpy()
    
    # Tag in chunks to avoid memory issues
    chunk_size = 50000
    pos_sequences = []
    t0 = time.time()
    
    for start in range(0, len(use_tokens), chunk_size):
        end = min(start + chunk_size, len(use_tokens))
        chunk = use_tokens[start:end].tolist()
        text = tok.decode(chunk)
        
        # Split into sentences and tag
        sentences = text.split('.')
        for sent in sentences:
            if len(sent.strip()) < 2:
                continue
            try:
                words = nltk.word_tokenize(sent)
                tagged = nltk.pos_tag(words, tagset='universal')
                pos_sequences.append([tag for _, tag in tagged])
            except Exception:
                pass
        
        if start % 200000 == 0:
            elapsed = time.time() - t0
            rate = end / max(elapsed, 1)
            print(f'    {end}/{len(use_tokens)} ({rate:.0f} tok/s)', flush=True)
    
    # Compile transition counts
    print(f'  Compiling POS transitions...')
    unigram_counts = Counter()
    bigram_counts = Counter()
    trigram_counts = Counter()
    
    for seq in pos_sequences:
        for tag in seq:
            unigram_counts[tag] += 1
        for i in range(len(seq) - 1):
            bigram_counts[(seq[i], seq[i+1])] += 1
        for i in range(len(seq) - 2):
            trigram_counts[(seq[i], seq[i+1], seq[i+2])] += 1
    
    # Build transition matrices
    tags = sorted(unigram_counts.keys())
    tag_to_idx = {t: i for i, t in enumerate(tags)}
    n_tags = len(tags)
    print(f'  {n_tags} POS tags: {tags}')
    
    bigram_trans = np.zeros((n_tags, n_tags), dtype=np.float32)
    trigram_trans = np.zeros((n_tags, n_tags, n_tags), dtype=np.float32)
    
    uni_total = sum(unigram_counts.values())
    for (t1, t2), cnt in bigram_counts.items():
        i, j = tag_to_idx[t1], tag_to_idx[t2]
        bigram_trans[i, j] = cnt / max(unigram_counts[t1], 1)
    
    for (t1, t2, t3), cnt in trigram_counts.items():
        i, j, k = tag_to_idx[t1], tag_to_idx[t2], tag_to_idx[t3]
        trigram_trans[i, j, k] = cnt / max(bigram_counts.get((t1, t2), 1), 1)
    
    return {
        'tags': tags,
        'tag_to_idx': tag_to_idx,
        'bigram_trans': bigram_trans,
        'trigram_trans': trigram_trans,
        'unigram_freq': np.array([unigram_counts[t] / uni_total for t in tags], dtype=np.float32),
    }


def main():
    out_dir = DEEPSEEK / 'artifacts/compiled_priors_v3'
    out_dir.mkdir(parents=True, exist_ok=True)
    V = 50257

    print('=' * 60)
    print(' BUILDING V3 COMPILED PRIORS')
    print('=' * 60)

    print('[1] Loading training data...')
    train_ids = torch.load(
        DEEPSEEK / 'artifacts/wikitext_gpt2/train_ids.pt',
        weights_only=False).long()
    print(f'  {len(train_ids):,} tokens')

    print('[2] Building sparse PPMI stats...')
    ppmi_stats = build_ppmi_stats(train_ids, V, max_tokens=500000)
    print(f'  {len(ppmi_stats["ppmi"]):,} PPMI pairs')

    print('[3] Building word-topic matrix...')
    t0 = time.time()
    word_topics = build_topic_from_sparse(ppmi_stats, V, K=50)
    torch.save(word_topics, out_dir / 'word_topics.pt')
    print(f'  Saved word_topics ({word_topics.shape}) in {time.time()-t0:.1f}s')

    print('[4] Building POS transitions...')
    t0 = time.time()
    pos_stats = build_pos_transitions(train_ids, V, max_tokens=2000000)
    with open(out_dir / 'pos_stats.pkl', 'wb') as f:
        pickle.dump(pos_stats, f)
    print(f'  Saved POS stats ({pos_stats["tags"]}) in {time.time()-t0:.1f}s')

    print(f'\nDone. Priors saved to {out_dir}')


if __name__ == '__main__':
    main()
