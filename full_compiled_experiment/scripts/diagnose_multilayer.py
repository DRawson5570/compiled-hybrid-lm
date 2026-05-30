"""
Multi-layer fidelity diagnostic. Tests every component of the UCN execution
chain against the real Qwen2.5-1.5B model to find bugs causing fidelity collapse.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"

print("Loading Qwen2.5-1.5B (fp32, eager)...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B", trust_remote_code=True,
    torch_dtype=torch.float32, attn_implementation="eager",
).to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)

prompt = "The cat sat on the mat and looked around with curiosity."
inp = tokenizer(prompt, return_tensors="pt").to(device)

# Collect all intermediate states from real model
real_attn_out = {}
real_mlp_out = {}
real_attn_input = {}  # pre-norm hidden state
real_mlp_input = {}    # post-attn-norm hidden state
real_hidden = {}       # post-layer hidden state

def make_hooks(l):
    def attn_pre(module, args, kwargs):
        if args: real_attn_input[l] = args[0].detach().clone()
        elif 'hidden_states' in kwargs: real_attn_input[l] = kwargs['hidden_states'].detach().clone()
    def attn_post(module, args, output):
        real_attn_out[l] = (output[0] if isinstance(output, tuple) else output).detach().clone()
    def mlp_pre(module, args, kwargs):
        if args: real_mlp_input[l] = args[0].detach().clone()
        elif 'hidden_states' in kwargs: real_mlp_input[l] = kwargs['hidden_states'].detach().clone()
    def mlp_post(module, args, output):
        real_mlp_out[l] = (output[0] if isinstance(output, tuple) else output).detach().clone()
    def layer_post(module, args, output):
        real_hidden[l] = (output[0] if isinstance(output, tuple) else output).detach().clone()
    return attn_pre, attn_post, mlp_pre, mlp_post, layer_post

target = [0, 4, 8]
all_hooks = []
for l in target:
    layer = model.model.layers[l]
    ap_pre, ap_post, mp_pre, mp_post, lp = make_hooks(l)
    all_hooks.append(layer.self_attn.register_forward_pre_hook(ap_pre, with_kwargs=True))
    all_hooks.append(layer.self_attn.register_forward_hook(ap_post))
    all_hooks.append(layer.mlp.register_forward_pre_hook(mp_pre, with_kwargs=True))
    all_hooks.append(layer.mlp.register_forward_hook(mp_post))
    all_hooks.append(layer.register_forward_hook(lp))

with torch.no_grad():
    out = model(**inp, output_hidden_states=True)
    emb = out.hidden_states[0].clone()

for h in all_hooks:
    h.remove()

print(f"\n{'='*60}")
print(f"TEST 1: RMS Norm Correctness")
print(f"{'='*60}")
from ucn.runtime.executor import rms_norm

for l in target:
    ln_in = model.model.layers[l].input_layernorm.weight.data.clone()
    ln_mlp = model.model.layers[l].post_attention_layernorm.weight.data.clone()

    # Pre-attention norm
    if l == 0:
        h_before = emb.clone()
    else:
        h_before = real_hidden[target[target.index(l)-1]].clone() if target.index(l) > 0 else emb.clone()

    our_rms = rms_norm(h_before, ln_in)
    real_rms = model.model.layers[l].input_layernorm(h_before)
    cos = F.cosine_similarity(our_rms.reshape(-1), real_rms.reshape(-1), dim=0).item()
    max_e = (our_rms - real_rms).abs().max().item()
    print(f"  Layer {l} pre-attn norm: cos={cos:.8f}, max_err={max_e:.6e}  {'OK' if cos>0.9999 else 'BUG'}")

    # Post-attention norm (after residual)
    h_after_attn = h_before + real_attn_out[l]
    our_rms2 = rms_norm(h_after_attn, ln_mlp)
    real_rms2 = model.model.layers[l].post_attention_layernorm(h_after_attn)
    cos2 = F.cosine_similarity(our_rms2.reshape(-1), real_rms2.reshape(-1), dim=0).item()
    max_e2 = (our_rms2 - real_rms2).abs().max().item()
    print(f"  Layer {l} post-attn norm: cos={cos2:.8f}, max_err={max_e2:.6e}  {'OK' if cos2>0.9999 else 'BUG'}")

print(f"\n{'='*60}")
print(f"TEST 2: Attention Output Fidelity Per Layer")
print(f"{'='*60}")
from ucn.dsl.ast import MatrixRef, Program, Transform
from ucn.backend.codegen.reference import ReferenceBackend

for l in target:
    a = model.model.layers[l].self_attn
    stdlib = {"a": {
        "operator_type": "multihead_attention",
        "W_q": a.q_proj.weight.float(), "b_q": a.q_proj.bias.float() if a.q_proj.bias is not None else None,
        "W_k": a.k_proj.weight.float(), "b_k": a.k_proj.bias.float() if a.k_proj.bias is not None else None,
        "W_v": a.v_proj.weight.float(), "b_v": a.v_proj.bias.float() if a.v_proj.bias is not None else None,
        "W_o": a.o_proj.weight.float(),
        "n_heads": 12, "n_kv_heads": 2, "head_dim": 128,
    }}
    ref = ReferenceBackend(stdlib_weights=stdlib, device="cpu", dtype=torch.float32)
    prog = Program(); prog.add_stmt("y", Transform("x", MatrixRef("stdlib", "a")))

    # Give the UCN attention the SAME input the real attention received
    real_input = real_attn_input[l].cpu()
    ucn_attn_out = ref.execute(prog, {"x": real_input})["y"]

    cos = F.cosine_similarity(ucn_attn_out.reshape(-1), real_attn_out[l].cpu().reshape(-1), dim=0).item()
    print(f"  Layer {l} attention: cos={cos:.8f}  {'OK' if cos>0.9999 else 'BUG!'}")

print(f"\n{'='*60}")
print(f"TEST 3: Norm Weight Identity Check")
print(f"{'='*60}")
for l in target:
    ln_in = model.model.layers[l].input_layernorm.weight.data
    ln_mlp = model.model.layers[l].post_attention_layernorm.weight.data
    print(f"  Layer {l} pre_attn norm weight: mean={ln_in.mean():.4f}, std={ln_in.std():.4f}, shape={list(ln_in.shape)}")
    print(f"  Layer {l} post_attn norm weight: mean={ln_mlp.mean():.4f}, std={ln_mlp.std():.4f}, shape={list(ln_mlp.shape)}")
    # Verify they're different (not same weight copied)
    cos_norm = F.cosine_similarity(ln_in.reshape(-1), ln_mlp.reshape(-1), dim=0).item()
    print(f"    pre_attn vs post_attn cosine: {cos_norm:.4f} {'(different—OK)' if cos_norm < 0.99 else '(SAME—SUSPICIOUS)'}")

print(f"\n{'='*60}")
print(f"TEST 4: MLP Input Correctness")
print(f"{'='*60}")
for l in target:
    # The MLP input should be: post_attention_layernorm(h + attention_output)
    h_before = real_attn_input[l]
    h_after_attn = h_before + real_attn_out[l]
    expected_mlp_input = model.model.layers[l].post_attention_layernorm(h_after_attn)
    actual_mlp_input = real_mlp_input[l]

    if actual_mlp_input is not None:
        cos = F.cosine_similarity(expected_mlp_input.reshape(-1), actual_mlp_input.reshape(-1), dim=0).item()
        print(f"  Layer {l} MLP input: cos={cos:.8f}  {'OK' if cos>0.9999 else 'DISCREPANCY'}")

print(f"\n{'='*60}")
print(f"TEST 5: Single-Layer UCN vs Real Per-Layer")
print(f"{'='*60}")
for l in target:
    # Run UCN for this layer ONLY, using the REAL model's hidden state as input
    # This tests whether the UCN layer components (attention + sparse MLP) work
    # when given correct inputs, isolating the multi-layer propagation issue.
    
    a = model.model.layers[l].self_attn
    from ucn.decompile.mlp_decomposer import extract_full_mlp_weights_lr

    stdlib = {
        "attn": {
            "operator_type": "multihead_attention",
            "W_q": a.q_proj.weight.float(), "b_q": a.q_proj.bias.float() if a.q_proj.bias is not None else None,
            "W_k": a.k_proj.weight.float(), "b_k": a.k_proj.bias.float() if a.k_proj.bias is not None else None,
            "W_v": a.v_proj.weight.float(), "b_v": a.v_proj.bias.float() if a.v_proj.bias is not None else None,
            "W_o": a.o_proj.weight.float(),
            "n_heads": 12, "n_kv_heads": 2, "head_dim": 128,
        }
    }
    mw = extract_full_mlp_weights_lr(model, l, rank=128)
    stdlib["mlp"] = {
        "operator_type": "sparse_down_projection_lr",
        "gate_weight": mw["gate_weight"], "gate_bias": mw["gate_bias"],
        "up_weight": mw["up_weight"], "down_weight": mw["down_weight"],
        "U_r": mw["U_r"], "V_r": mw["V_r"], "top_k": 1024,
    }

    ref = ReferenceBackend(stdlib_weights=stdlib, device="cpu", dtype=torch.float32)
    ln_in = model.model.layers[l].input_layernorm.weight.data.clone().cpu()
    ln_mlp = model.model.layers[l].post_attention_layernorm.weight.data.clone().cpu()

    # Use the REAL hidden state before this layer as input
    h_input = real_attn_input[l].cpu() if real_attn_input[l] is not None else emb.cpu()

    # Attention
    hn = rms_norm(h_input, ln_in)
    pa = Program(); pa.add_stmt("y", Transform("x", MatrixRef("stdlib", "attn")))
    ucn_attn = ref.execute(pa, {"x": hn})["y"]
    h_after = h_input + ucn_attn

    # MLP
    hn2 = rms_norm(h_after, ln_mlp)
    pm = Program(); pm.add_stmt("y", Transform("x", MatrixRef("stdlib", "mlp")))
    ucn_mlp = ref.execute(pm, {"x": hn2})["y"]
    h_final = h_after + ucn_mlp

    cos = F.cosine_similarity(h_final.reshape(-1), real_hidden[l].cpu().reshape(-1), dim=0).item()
    print(f"  Layer {l} single-layer UCN: cos={cos:.8f}")

print(f"\n{'='*60}")
print(f"TEST 6: Sparse MLP Fidelity Per Layer")
print(f"{'='*60}")
for l in target:
    mw = extract_full_mlp_weights_lr(model, l, rank=128)
    stdlib = {"mlp": {
        "operator_type": "sparse_down_projection_lr",
        "gate_weight": mw["gate_weight"], "gate_bias": mw["gate_bias"],
        "up_weight": mw["up_weight"], "down_weight": mw["down_weight"],
        "U_r": mw["U_r"], "V_r": mw["V_r"], "top_k": 1024,
    }}
    ref = ReferenceBackend(stdlib_weights=stdlib, device="cpu", dtype=torch.float32)
    prog = Program(); prog.add_stmt("y", Transform("x", MatrixRef("stdlib", "mlp")))

    mlp_in = real_mlp_input[l].cpu() if real_mlp_input[l] is not None else None
    mlp_out_real = real_mlp_out[l].cpu() if real_mlp_out[l] is not None else None

    if mlp_in is not None and mlp_out_real is not None:
        ucn_out = ref.execute(prog, {"x": mlp_in})["y"]
        cos = F.cosine_similarity(ucn_out.reshape(-1), mlp_out_real.reshape(-1), dim=0).item()
        print(f"  Layer {l} sparse MLP: cos={cos:.8f}")

print(f"\n{'='*60}")
print(f"TEST 7: Residual Connection Magnitudes")
print(f"{'='*60}")
for l in target:
    h_in = real_attn_input[l] if real_attn_input[l] is not None else emb
    attn_norm = real_attn_out[l].norm(p=2) / h_in.norm(p=2)
    real_mlp_norm = real_mlp_out[l].norm(p=2) / (h_in + real_attn_out[l]).norm(p=2)
    print(f"  Layer {l}: attn/h ratio={attn_norm:.6f}, mlp/h ratio={real_mlp_norm:.6f}")

print(f"\n{'='*60}")
print(f"VERDICT")
print(f"{'='*60}")
print("  If Tests 1-4 all show OK (>0.9999): the per-component infrastructure is correct.")
print("  If Test 5 shows 0.80 at layer 0 but lower at layers 4,8: sparse MLP fidelity varies by layer.")
print("  If Test 5 is consistent across layers: the bug is in multi-layer propagation (how hidden state feeds between steps).")
print("  If Test 6 drops across layers: sparse MLP quality degrades in deeper layers.")
