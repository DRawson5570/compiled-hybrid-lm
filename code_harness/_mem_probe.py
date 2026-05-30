"""Probe: 4-bit base train-step peak memory, gradient checkpointing ON vs OFF."""
import sys, gc
import torch
import torch.nn.functional as F
from train_rft import RFTTrainer, humaneval_problems

MODE = sys.argv[1] if len(sys.argv) > 1 else "gc"

t = RFTTrainer(device="cuda:0", seq_cap=512, load_4bit=True)
if MODE == "nogc":
    t.model.gradient_checkpointing_disable()
    print("MODE: GC OFF", flush=True)
else:
    t.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print("MODE:", MODE, flush=True)

opt = torch.optim.AdamW(t.steerer.parameters(), lr=3e-4)
t.set_active(True)
t.steerer.train()

# Warm an eval to leave realistic recurrent-state high-water mark.
t.eval_humaneval(humaneval_problems()[:8], active=True, batch=1)

if MODE == "gc_train":
    # Put the base in train() so the `self.gradient_checkpointing and self.training`
    # gate in each decoder layer actually engages. Params stay frozen.
    t.model.train()
    print("base set to train() so GC gate engages", flush=True)

vocab = t.model.config.vocab_size
for L in [256, 320, 384, 448, 512]:
    try:
        torch.cuda.reset_peak_memory_stats()
        x = torch.randint(0, vocab, (1, L), device="cuda:0")
        y = torch.randint(0, vocab, (1, L), device="cuda:0")
        start = 8  # short prefix -> most positions supervised (worst case)
        opt.zero_grad()
        hidden = t.model.model(input_ids=x).last_hidden_state[:, start:, :]
        logits = t.model.lm_head(hidden).float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y[:, start:].reshape(-1))
        loss = loss + 0.00005 * t.steerer.orthogonal_penalty()
        loss.backward()
        opt.step()
        print("effL=%d peak=%.2fGB" % (L, torch.cuda.max_memory_allocated() / 1e9), flush=True)
        del hidden, logits, loss, x, y
    except torch.OutOfMemoryError:
        print("effL=%d -> OOM" % L, flush=True)
    gc.collect()
    torch.cuda.empty_cache()
print("done", flush=True)
