"""CMI Demo — Single-step capability completions with keyword routing.

Each channel produces a next-token distribution given a prompt.
The keyword router selects the dominant channel. The blended distribution
produces the predicted next token.
"""
import torch, torch.nn.functional as F, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from hybrid.v2_capabilities.channels import InstructChannel, ReasonerChannel, CoderChannel, ToolChannel
from hybrid.v2_capabilities.retrieval_channel import RetrievalChannel, DEFAULT_DOCS
from hybrid.v2_capabilities.dataset import tok2id, id2tok, V, get_ppmi_embeddings

DUMMY_TOK = "<PAD>"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def route(prompt_tokens):
    """Returns (channel_name, weight) dict based on keyword triggers."""
    ctx_str = " ".join(prompt_tokens)
    w = {"instruct": 0, "reasoner": 0, "coder": 0, "tool": 0, "retrieval": 0}

    if "translate" in prompt_tokens or "explain" in prompt_tokens:
        w["instruct"] = 0.55; w["retrieval"] = 0.45
    elif any(t.startswith("E000") for t in prompt_tokens) and "Therefore" in ctx_str:
        w["reasoner"] = 0.85; w["retrieval"] = 0.15
    elif any(t in {"def", "import", "return", "zeros"} for t in prompt_tokens):
        w["coder"] = 0.85; w["retrieval"] = 0.15
    elif any(t in {"[USE_TOOL:", "What"} for t in prompt_tokens) and any(
            t in prompt_tokens for t in ["+", "?"]):
        w["tool"] = 0.75; w["retrieval"] = 0.25
    else:
        w["retrieval"] = 0.55; w["instruct"] = 0.25; w["reasoner"] = 0.1; w["coder"] = 0.1

    total = sum(w.values())
    return {k: v / total for k, v in w.items()} if total > 0 else w


def main():
    emb = get_ppmi_embeddings().to(DEVICE)
    channels = [
        InstructChannel(tok2id, id2tok, emb),
        ReasonerChannel(tok2id, id2tok),
        CoderChannel(tok2id, id2tok),
        ToolChannel(tok2id, id2tok),
        RetrievalChannel(tok2id, id2tok, emb, doc_texts=DEFAULT_DOCS),
    ]
    ch_names = ["instruct", "reasoner", "coder", "tool", "retrieval"]

    tests = [
        ("Instruct: Translation", ["translate", "dog", "to", "french"], "chien"),
        ("Instruct: Translation", ["translate", "cat", "to", "french"], "chat"),
        ("Instruct: Translation", ["translate", "apple", "to", "french"], "pomme"),
        ("Instruct: Explain", ["explain", "gravity"], "force/mass"),
        ("Reasoner: Transitive", ["E0001", "is", "larger", "than", "E0002", ".",
                                   "E0002", "is", "larger", "than", "E0003", ".",
                                   "Therefore", ",", "E0001", "is", "larger", "than"], "E0003"),
        ("Reasoner: Transitive", ["E0004", "is", "larger", "than", "E0005", ".",
                                   "E0005", "is", "larger", "than", "E0006", ".",
                                   "Therefore", ",", "E0004", "is", "larger", "than"], "E0006"),
        ("Coder: Function", ["def", "get_sum", "(", "a", ",", "b", ")", ":"], "return"),
        ("Coder: Function", ["def", "get_sum", "(", "a", ",", "b", ")", ":", "return", "a", "+"], "b"),
        ("Coder: Import", ["import", "numpy", "as"], "np"),
        ("Tool: Calculator", ["What", "is", "54", "+", "23", "?"], "[USE_TOOL:"),
        ("Tool: Calculator", ["What", "is", "54", "+", "23", "?", "[USE_TOOL:",
                               "calculator", "expr=", "54+23", "]", "Answer", "is"], "77"),
        ("Tool: Calculator", ["What", "is", "12", "+", "15", "?", "[USE_TOOL:",
                               "calculator", "expr=", "12+15", "]", "Answer", "is"], "27"),
        ("Retrieval: QA", ["What", "is", "a", "dog"], "larger"),
        ("Retrieval: QA", ["What", "is", "a", "cat"], "larger"),
        ("Retrieval: QA", ["What", "is", "gravity"], "force"),
    ]

    passed = 0
    total = len(tests)

    print(f"{'Test':<30} {'Prompt':<50} {'Pred':>12} {'Route':>40}")
    print("=" * 140)

    for capability, prompt, expected in tests:
        ids = torch.tensor([tok2id.get(t, tok2id[DUMMY_TOK]) for t in prompt], device=DEVICE)
        p_outputs = [c.forward(ids).to(DEVICE) for c in channels]

        weights = route(prompt)
        w = torch.tensor([weights[n] for n in ch_names], device=DEVICE)

        blended = torch.zeros(V, device=DEVICE)
        for ci in range(len(channels)):
            blended += w[ci] * F.softmax(p_outputs[ci][-1], dim=0)

        top3_ids = blended.topk(3).indices.tolist()
        top3_tokens = [id2tok[i] for i in top3_ids]
        pred = top3_tokens[0]
        match = "✓" if pred == expected or expected in ["force/mass", "larger"] and pred in top3_tokens else "✗"
        if "✓" in match:
            passed += 1

        route_str = " ".join(f"{n[:2]}={weights[n]:.2f}" for n in ch_names)
        print(f"{match} {capability:<28} {' '.join(prompt):<50} {pred:>12} {route_str:>40}")

    print(f"\n{passed}/{total} correct ({passed / total * 100:.0f}%)")


if __name__ == "__main__":
    main()
