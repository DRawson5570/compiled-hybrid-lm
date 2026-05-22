"""CMI Demo — 5-channel Compiled Modular Intelligence with keyword routing.

Uses explicit trigger-based routing (no learned blender needed at demo scale).
Each capability channel fires when its trigger keywords are detected.
"""
import torch, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from hybrid.v2_capabilities.channels import InstructChannel, ReasonerChannel, CoderChannel, ToolChannel
from hybrid.v2_capabilities.retrieval_channel import RetrievalChannel, DEFAULT_DOCS
from hybrid.v2_capabilities.dataset import tok2id, id2tok, V, get_ppmi_embeddings

DUMMY_TOK = "<PAD>"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def route_channels(context_tokens: list[str]):
    """Returns (channel_name, weight) pairs based on keyword triggers."""
    ctx_str = " ".join(context_tokens)

    weights = {
        "instruct": 0.0,
        "reasoner": 0.0,
        "coder": 0.0,
        "tool": 0.0,
        "retrieval": 0.0,
    }

    # Instruct triggers: translation, explanation
    if "translate" in context_tokens:
        weights["instruct"] = 0.6
        weights["retrieval"] = 0.4
    elif "explain" in context_tokens:
        weights["instruct"] = 0.6
        weights["retrieval"] = 0.4
    # Reasoner triggers: entity comparison chains
    elif any(t.startswith("E000") for t in context_tokens):
        weights["reasoner"] = 0.8
        weights["retrieval"] = 0.2
    # Coder triggers: def, import, function syntax
    elif any(t in {"def", "import", "return", "zeros"} for t in context_tokens):
        weights["coder"] = 0.8
        weights["retrieval"] = 0.2
    # Tool triggers: calculator
    elif any(t in {"[USE_TOOL:", "What"} for t in context_tokens) and any(
        t in context_tokens for t in ["+", "?"]
    ):
        weights["tool"] = 0.7
        weights["retrieval"] = 0.3
    # Default: retrieval-dominant (question answering)
    else:
        weights["retrieval"] = 0.5
        weights["instruct"] = 0.3
        weights["reasoner"] = 0.1
        weights["coder"] = 0.1

    # Normalize
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}

    return weights


def generate(channels, emb, prompt: list[str], max_tokens: int = 12):
    """Auto-regressive generation with keyword-triggered routing."""
    current = list(prompt)
    channel_list = [c for c in channels]

    print(f"\n{'─'*60}")
    prompt_str = " ".join(prompt)
    print(f"Prompt: {prompt_str}")
    print(f"{'Step':>4} {'Token':>14} {'Route':>45}")
    print(f"{'─'*60}")

    for step in range(max_tokens):
        ids = torch.tensor([tok2id.get(t, tok2id[DUMMY_TOK]) for t in current], device=DEVICE)

        # Get weights from keyword routing
        weights = route_channels(current)
        w = torch.tensor([weights[n] for n in
                          ["instruct", "reasoner", "coder", "tool", "retrieval"]],
                         device=DEVICE)

        # Run all channels and blend
        p_outputs = [c.forward(ids).to(DEVICE) for c in channel_list]
        blended = torch.zeros(V, device=DEVICE)
        for ci in range(len(channel_list)):
            blended += w[ci] * p_outputs[ci][-1].exp()

        next_id = blended.argmax().item()
        next_tok = id2tok[next_id]
        current.append(next_tok)

        # Show routing
        dom_ch = ["instruct", "reasoner", "coder", "tool", "retrieval"][w.argmax().item()]
        w_str = " ".join(
            f"{['I','R','C','T','Rt'][i]}={w[i].item():.2f}" for i in range(5)
        )
        print(f"{step:4d} {next_tok:>14} {dom_ch:>10} [{w_str}]")

    result = " ".join(current[len(prompt):])
    print(f"Result: {result}")
    return result


def main():
    emb = get_ppmi_embeddings().to(DEVICE)

    channels = [
        InstructChannel(tok2id, id2tok, emb),
        ReasonerChannel(tok2id, id2tok),
        CoderChannel(tok2id, id2tok),
        ToolChannel(tok2id, id2tok),
        RetrievalChannel(tok2id, id2tok, emb, doc_texts=DEFAULT_DOCS),
    ]

    print(f"CMI Demo — 5 Channels with Keyword Routing")
    print(f"Vocabulary: {V} tokens, Device: {DEVICE}")
    print(f"Channels: Instruct, Reasoner, Coder, Tool, Retrieval")

    # Test each capability
    tests = [
        # Instruct: translation
        (["translate", "dog", "to", "french"], "→ chien"),
        (["translate", "cat", "to", "french"], "→ chat"),
        (["explain", "gravity"], "→ force, mass..."),
        # Reasoner: transitive chain
        (["E0001", "is", "larger", "than", "E0002", ".",
          "E0002", "is", "larger", "than", "E0003", ".",
          "Therefore", ",", "E0001", "is", "larger", "than"], "→ E0003"),
        # Coder: function definition
        (["def", "get_sum", "(", "a", ",", "b", ")", ":", "return", "a", "+"], "→ b"),
        (["import", "numpy", "as", "np", ".", "zeros", "("], "→ 10"),
        # Tool: calculator
        (["What", "is", "54", "+", "23", "?", "[USE_TOOL:",
          "calculator", "expr=", "54+23", "]", "Answer", "is"], "→ 77"),
        # Retrieval: document Q&A
        (["What", "is", "a", "dog"], "→ larger (from docs)"),
        (["What", "is", "gravity"], "→ force (from docs)"),
    ]

    for prompt, expected in tests:
        generate(channels, emb, prompt)


if __name__ == "__main__":
    main()
