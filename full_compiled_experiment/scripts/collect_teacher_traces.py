"""
Gap C: Teacher trace collector.

Runs Qwen2.5-1.5B on prompts, collects (embeddings, template_id, params)
distillation targets for training a MetaCompiler.
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ucn.dsl.ast import Activate, ActivateType, MatrixRef, Mix, Program, Transform
from ucn.backend.codegen.reference import ReferenceBackend
from ucn.frontend.template_library import TemplateLibrary


def collect_traces(executor, hidden_states, templates, stdlib_names, layer_idx):
    traces = []
    lib = TemplateLibrary()

    h = hidden_states.to(dtype=torch.float32)
    B, T, D = h.shape
    h_flat = h.reshape(-1, D)

    best_cos = -1.0
    best_program = None
    best_template = 0
    best_params = []

    for tid in range(lib.n_templates):
        for _ in range(5):
            params = [torch.rand(1).item() for _ in range(lib.templates[tid].n_params)]
            program = lib.build_program(tid, params, stdlib_names)

            try:
                outputs = executor.execute_raw(program, {"x": h_flat}, batch_size=h_flat.shape[0])
                y = outputs.get("y", torch.zeros_like(h_flat))
                cos = float(F.cosine_similarity(y.reshape(-1), h_flat.reshape(-1), dim=0).item())
            except Exception:
                cos = -1.0

            if cos > best_cos:
                best_cos = cos
                best_program = program
                best_template = tid
                best_params = params

    traces.append((h, best_template, torch.tensor(best_params), best_cos))
    return traces


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading Qwen2.5-1.5B...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B", trust_remote_code=True,
        torch_dtype=torch.float32, attn_implementation="eager",
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)

    prompts = [
        "The cat sat on the mat and looked around with great curiosity.",
        "Machine learning is a field of artificial intelligence.",
        "Python is a high-level programming language for data science.",
        "The quick brown fox jumps over the lazy dog near the river.",
        "Neural networks consist of interconnected layers of nodes.",
        "The Earth orbits the Sun at about 93 million miles distance.",
        "Water boils at 100 degrees Celsius under standard pressure.",
        "Shakespeare wrote many famous plays in the English language.",
        "Deep learning models require large amounts of training data.",
        "The speed of light in a vacuum is approximately 299792458 m/s.",
    ]

    target_layers = [0, 8]
    hidden = {l: [] for l in target_layers}

    hooks = []
    for l in target_layers:
        def hk(l):
            def hook(module, input, output):
                hidden[l].append((output[0] if isinstance(output, tuple) else output).detach().cpu())
            return hook
        hooks.append(model.model.layers[l].register_forward_hook(hk(l)))

    print(f"Running {len(prompts)} prompts...")
    for prompt in prompts:
        inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32).to(device)
        with torch.no_grad():
            model(**inp)

    for h in hooks:
        h.remove()

    stdlib_names = ["primitive_0", "primitive_1", "primitive_2", "primitive_3"]

    simple_stdlib = {n: {"operator_type": "low_rank_projection",
                         "u": torch.randn(8, 1536) * 0.01,
                         "v": torch.randn(8, 1536) * 0.01} for n in stdlib_names}

    executor = ReferenceBackend(stdlib_weights=simple_stdlib, device="cpu", dtype=torch.float32)

    all_traces = []
    for l in target_layers:
        hs = torch.cat(hidden[l], dim=0)
        traces = collect_traces(executor, hs, None, stdlib_names, l)
        all_traces.extend(traces)
        avg_cos = sum(t[3] for t in traces) / len(traces) if traces else 0
        print(f"  Layer {l}: {len(traces)} traces, avg best cosine={avg_cos:.4f}")

    out_dir = Path(__file__).resolve().parent.parent / "artifacts" / "teacher_traces"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "traces": [(t[0], t[1], t[2]) for t in all_traces],
        "stdlib_names": stdlib_names,
        "layers": target_layers,
    }, out_dir / "distillation_data.pt")

    print(f"Saved {len(all_traces)} traces to {out_dir / 'distillation_data.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
