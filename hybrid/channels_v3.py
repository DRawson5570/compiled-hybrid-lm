"""channels_v3.py — Full compiled channel features with Gemini V3.1 upgrades.

Channel inventory (21 total):
  0-14: Base 15 (n-grams, recency, entropy, PPMI stubs, shape)
  15-17: Register (punct_density, repetition_score, unique_token_ratio)
  18: Topic log-probability
  19: KV retrieval max similarity
  20: POS bigram transition log-probability

Includes Witten-Bell smoothing for sharper unseen handling.
"""
import numpy as np
import math
import torch
from collections import defaultdict

from hybrid.smoothing import WittenBellSmoother


class FullV3ChannelFeatures:
    """21-channel compiled prior with Witten-Bell, topic vector, KV cache, POS."""

    def __init__(self, V=50257, punct_ids=None, word_topics=None,
                 pos_stats=None, ppmi_embeddings=None):
        self.V = V
        self._punct_ids = punct_ids or set()
        self._uniform = -math.log(V)

        # Base caches
        self._uni = np.zeros(V, dtype=np.float32)
        self._bi = {}; self._bit = {}
        self._tri = {}; self._trit = {}
        self._skip2 = {}; self._skip2t = {}
        self._skip3 = {}; self._skip3t = {}
        self._ctx = []; self._step = 0
        self._seen = defaultdict(list)

        # V3: register tracking
        self._punct_window = []
        self._token_window = []
        self._window_size = 128

        # Witten-Bell smoothers
        self._wb_uni = WittenBellSmoother(V)
        self._wb_bi = WittenBellSmoother(V)
        self._wb_tri = WittenBellSmoother(V)

        # Topic vector (set externally)
        self._word_topics = word_topics  # (V, K)
        self._topic_vec = None
        if word_topics is not None:
            K = word_topics.shape[1]
            self._topic_vec = np.zeros(K, dtype=np.float32)
            self._topic_lambda = 0.95

        # POS tracker
        self._pos_stats = pos_stats  # dict with tags, bigram_trans, etc.
        self._pos_ctx = []  # recent POS tags

        # KV semantic cache
        self._ppmi_embeddings = ppmi_embeddings  # (V, d)
        self._kv_keys = None
        self._kv_vals = None
        self._kv_norms = None
        self._kv_count = 0
        self._kv_ptr = 0
        self._kv_max = 128
        if ppmi_embeddings is not None:
            d = ppmi_embeddings.shape[1]
            self._kv_keys = np.zeros((self._kv_max, d), dtype=np.float32)
            self._kv_vals = np.zeros(self._kv_max, dtype=np.int64)
            self._kv_norms = np.zeros(self._kv_max, dtype=np.float32)

    def update(self, token):
        tid = int(token)
        self._step += 1
        self._ctx.append(tid)
        self._ctx = self._ctx[-128:]

        if self._step % 10 == 0:
            self._uni *= 0.999
        if tid < self.V:
            self._uni[tid] += 1
            self._wb_uni.add(tid)

        if len(self._ctx) >= 2:
            p, c = self._ctx[-2], self._ctx[-1]
            self._bi[(p, c)] = self._bi.get((p, c), 0) + 1
            self._bit[p] = self._bit.get(p, 0) + 1
            self._wb_bi.add((p, c))
        if len(self._ctx) >= 3:
            p2, p1, c = self._ctx[-3], self._ctx[-2], self._ctx[-1]
            self._tri[(p2, p1, c)] = self._tri.get((p2, p1, c), 0) + 1
            self._trit[(p2, p1)] = self._trit.get((p2, p1), 0) + 1
            self._wb_tri.add((p2, p1, c))
        if len(self._ctx) >= 2:
            p2 = self._ctx[-2]
            self._skip2[(p2, tid)] = self._skip2.get((p2, tid), 0) + 1
            self._skip2t[p2] = self._skip2t.get(p2, 0) + 1
        if len(self._ctx) >= 3:
            p3 = self._ctx[-3]
            self._skip3[(p3, tid)] = self._skip3.get((p3, tid), 0) + 1
            self._skip3t[p3] = self._skip3t.get(p3, 0) + 1

        self._seen[tid].append(self._step)

        # Register tracking
        self._punct_window.append(tid in self._punct_ids)
        if len(self._punct_window) > self._window_size:
            self._punct_window = self._punct_window[-self._window_size:]
        self._token_window.append(tid)
        if len(self._token_window) > self._window_size:
            self._token_window = self._token_window[-self._window_size:]

        # Topic vector update
        if self._topic_vec is not None and self._word_topics is not None:
            if tid < self.V:
                token_topic = self._word_topics[tid].numpy()
                self._topic_vec = (self._topic_lambda * self._topic_vec +
                                   (1 - self._topic_lambda) * token_topic)

        # POS tracking
        if self._pos_stats is not None:
            tag = self._pos_stats.get('token_to_tag', {}).get(tid, 'X')
            self._pos_ctx.append(tag)
            if len(self._pos_ctx) > 3:
                self._pos_ctx = self._pos_ctx[-3:]

        # KV cache update
        if self._kv_keys is not None and self._ppmi_embeddings is not None:
            if tid < self.V:
                emb = self._ppmi_embeddings[tid].float().numpy()
                self._kv_keys[self._kv_ptr] = emb
                self._kv_vals[self._kv_ptr] = tid
                self._kv_norms[self._kv_ptr] = np.linalg.norm(emb)
                self._kv_ptr = (self._kv_ptr + 1) % self._kv_max
                self._kv_count = min(self._kv_count + 1, self._kv_max)

    def get_features(self, target):
        tid = int(target); ctx = self._ctx; u = self._uniform; feats = []

        # 0: unigram (WB smoothed)
        d = self._uni.sum() + 0.001 * self.V
        ul = self._wb_uni.smooth(self._uni[tid], d, u) if d > 0 and tid < self.V else u
        feats.append(float(ul))

        # 1-2: bigram fast/slow (WB)
        if len(ctx) >= 1:
            prev = ctx[-1]; tot = self._bit.get(prev, 0)
            cnt = self._bi.get((prev, tid), 0)
            bl = self._wb_bi.smooth(cnt, tot, u) if tot > 0 else u
        else:
            bl = u
        feats.append(float(bl)); feats.append(float(bl))

        # 3-4: trigram fast/slow
        if len(ctx) >= 2:
            ck = (ctx[-2], ctx[-1]); tot = self._trit.get(ck, 0)
            cnt = self._tri.get((ctx[-2], ctx[-1], tid), 0)
            tl = self._wb_tri.smooth(cnt, tot, u) if tot > 0 else u
        else:
            tl = u
        feats.append(float(tl)); feats.append(float(tl))

        # 5: skip2
        if len(ctx) >= 2:
            tot = self._skip2t.get(ctx[-2], 0)
            cnt = self._skip2.get((ctx[-2], tid), 0)
            s2 = math.log(max((cnt + 0.001) / (tot + 0.001 * self.V), 1e-7)) if tot > 0 else u
        else:
            s2 = u
        feats.append(float(s2))

        # 6: skip3
        if len(ctx) >= 3:
            tot = self._skip3t.get(ctx[-3], 0)
            cnt = self._skip3.get((ctx[-3], tid), 0)
            s3 = math.log(max((cnt + 0.001) / (tot + 0.001 * self.V), 1e-7)) if tot > 0 else u
        else:
            s3 = u
        feats.append(float(s3))

        # 7: recency
        pos = self._seen.get(tid, []); gap = 128 if not pos else min(128, self._step - pos[-1])
        rl = math.log(max(1.0 / max(gap, 1), 1e-7))
        feats.append(float(rl))

        # 8: builder_entropy
        d = self._uni.sum() + 0.001 * self.V
        if d > 0:
            probs = (self._uni + 0.001) / d; valid = probs > 0
            entropy = -np.sum(probs[valid] * np.log(probs[valid]))
            ent = float(entropy / math.log(self.V)) if entropy > 0 else 1.0
        else:
            ent = 1.0
        feats.append(float(ent))

        # 9-14: shape, global_uni, PPMI stubs
        feats.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # 15: punct_density
        pd = float(sum(self._punct_window) / max(len(self._punct_window), 1))
        feats.append(pd)

        # 16: repetition_score
        if len(self._token_window) >= 4:
            rc = sum(1 for i in range(3, len(self._token_window))
                    if self._token_window[i] == self._token_window[i-1])
            rs = float(rc / max(len(self._token_window) - 3, 1))
        else:
            rs = 0.0
        feats.append(rs)

        # 17: unique_token_ratio
        ur = float(len(set(self._token_window)) / max(len(self._token_window), 1))
        feats.append(ur)

        # 18: Topic log-probability
        if self._topic_vec is not None and self._word_topics is not None and tid < self.V:
            topic_prior = self._topic_vec @ self._word_topics[tid].numpy()
            tp = math.log(max(float(topic_prior), 1e-12))
        else:
            tp = 0.0
        feats.append(tp)

        # 19: KV retrieval max similarity
        if self._kv_keys is not None and tid < self.V and self._kv_count > 0:
            emb = self._ppmi_embeddings[tid].float().numpy()
            tnorm = np.linalg.norm(emb)
            if tnorm > 1e-8:
                sims = (self._kv_keys[:self._kv_count] @ emb) / (
                    self._kv_norms[:self._kv_count] * tnorm + 1e-8)
                kv_max = float(np.max(np.clip(sims, -1, 1)))
            else:
                kv_max = 0.0
        else:
            kv_max = 0.0
        feats.append(kv_max)

        # 20: POS bigram transition
        if self._pos_stats is not None and len(self._pos_ctx) >= 1:
            tag_map = self._pos_stats['tag_to_idx']
            bigram = self._pos_stats['bigram_trans']
            target_tag = self._pos_stats.get('token_to_tag', {}).get(tid)
            prev_tag = self._pos_ctx[-1]
            if target_tag and prev_tag and target_tag in tag_map and prev_tag in tag_map:
                ti, tj = tag_map[target_tag], tag_map[prev_tag]
                pos_lp = math.log(max(float(bigram[ti, tj]), 1e-7))
            else:
                pos_lp = 0.0
        else:
            pos_lp = 0.0
        feats.append(float(pos_lp))

        return feats

    def reset(self):
        self._uni = np.zeros(self.V, dtype=np.float32)
        self._bi = {}; self._bit = {}
        self._tri = {}; self._trit = {}
        self._skip2 = {}; self._skip2t = {}
        self._skip3 = {}; self._skip3t = {}
        self._ctx = []; self._step = 0
        self._seen = defaultdict(list)
        self._punct_window = []; self._token_window = []
        self._wb_uni = WittenBellSmoother(self.V)
        self._wb_bi = WittenBellSmoother(self.V)
        self._wb_tri = WittenBellSmoother(self.V)
        if self._word_topics is not None:
            self._topic_vec = np.zeros(self._word_topics.shape[1], dtype=np.float32)
        self._pos_ctx = []
        if self._kv_keys is not None:
            self._kv_count = 0; self._kv_ptr = 0
