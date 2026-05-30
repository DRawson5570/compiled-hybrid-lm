"""Extended router training for ARC rack with no-op route."""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from hybrid.cartridge_harness.suites import build_all_suites


NONE_ROUTE_LABEL = "none"
ARC_CARTRIDGE_ID = "qwen-arc-challenge-cartridge"

NONE_EXAMPLES = [
    "What is the capital of France?",
    "Explain how photosynthesis works.",
    "Write a poem about the ocean.",
    "Summarize the plot of Hamlet.",
    "What are the causes of World War I?",
    "Describe the water cycle.",
    "Who invented the telephone?",
    "How does a car engine work?",
    "What is the difference between DNA and RNA?",
    "Explain the theory of relativity.",
    "What are the major religions of the world?",
    "How to bake a chocolate cake?",
    "Describe the solar system.",
    "What is blockchain technology?",
    "Explain the concept of supply and demand.",
    "Who wrote Romeo and Juliet?",
    "What is the Pythagorean theorem?",
    "How do vaccines work?",
    "Describe the process of mitosis.",
    "What is climate change?",
    "How to change a car tire?",
    "Explain the difference between weather and climate.",
    "What are the benefits of exercise?",
    "Describe how the internet works.",
    "What is machine learning?",
    "How to write a resume?",
    "Explain the carbon cycle.",
    "What are human rights?",
    "How does a microwave work?",
    "Describe the structure of an atom.",
]


def build_arc_router_examples(
    dataset_name: str = "allenai/ai2_arc",
    config: str = "ARC-Challenge",
    split: str = "train",
    max_examples: int = 200,
) -> list[str]:
    from datasets import load_dataset

    try:
        ds = load_dataset(dataset_name, config, trust_remote_code=True)
    except (TypeError, ValueError):
        ds = load_dataset(dataset_name, config)
    raw = [dict(item) for item in ds[split]]

    from hybrid.benchmarks.arc_prompts import get_template

    template = get_template("arc_v1")
    prompts: list[str] = []
    for item in raw[:max_examples]:
        question = item["question"]
        choices_raw = item["choices"]
        if isinstance(choices_raw, dict):
            labels = choices_raw.get("label", [])
            texts = choices_raw.get("text", [])
        else:
            labels = [c["label"] for c in choices_raw]
            texts = [c["text"] for c in choices_raw]
        choices_block = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
        prompt = f"Question: {question}\n\nChoices:\n{choices_block}\n\nAnswer:"
        prompts.append(prompt)
    return prompts


def train_arc_rack_router(
    *,
    model_name: str = "Qwen/Qwen2.5-1.5B",
    device: str = "cuda",
    out_dir: str | Path,
    epochs: int = 300,
    lr: float = 3e-3,
    arc_max_examples: int = 200,
    none_max_examples: int = 200,
    class_balanced_sampling: bool = True,
) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_device = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch_device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(torch_device)
    hf_model.eval()
    for param in hf_model.parameters():
        param.requires_grad = False

    suites = build_all_suites()
    old_ids = tuple(suite.cartridge_id for suite in suites)
    cartridge_ids = old_ids + (ARC_CARTRIDGE_ID, NONE_ROUTE_LABEL)
    label_by_id = {cid: idx for idx, cid in enumerate(cartridge_ids)}

    train_examples: list[tuple[str, int]] = []
    val_examples: list[tuple[str, int]] = []

    for suite in suites:
        label = label_by_id[suite.cartridge_id]
        for task in suite.tasks:
            target = train_examples if task.split == "train" else val_examples
            target.append((task.prompt, label))

    arc_prompts = build_arc_router_examples(max_examples=arc_max_examples)
    arc_label = label_by_id[ARC_CARTRIDGE_ID]
    random.seed(42)
    random.shuffle(arc_prompts)
    split_idx = max(1, int(len(arc_prompts) * 0.8))
    for p in arc_prompts[:split_idx]:
        train_examples.append((p, arc_label))
    for p in arc_prompts[split_idx:]:
        val_examples.append((p, arc_label))

    none_label = label_by_id[NONE_ROUTE_LABEL]
    none_prompts = NONE_EXAMPLES[:none_max_examples]
    random.shuffle(none_prompts)
    none_split = max(1, int(len(none_prompts) * 0.8))
    for p in none_prompts[:none_split]:
        train_examples.append((p, none_label))
    for p in none_prompts[none_split:]:
        val_examples.append((p, none_label))

    print(f"Router training examples: {len(train_examples)} train, {len(val_examples)} val", flush=True)
    print(f"  Label distribution (train): "
          f"{ {k: sum(1 for _, l in train_examples if l == v) for k, v in label_by_id.items()} }",
          flush=True)

    def encode(prompt: str) -> torch.Tensor:
        ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=256).to(torch_device)
        with torch.no_grad():
            out = hf_model(ids, output_hidden_states=True, use_cache=False)
            return out.hidden_states[-1][0].mean(dim=0).float().cpu()

    def materialize(examples: list[tuple[str, int]]):
        embeddings = torch.stack([encode(prompt) for prompt, _ in examples])
        labels = torch.tensor([label for _, label in examples], dtype=torch.long)
        return embeddings, labels

    train_x, train_y = materialize(train_examples)
    val_x, val_y = materialize(val_examples)

    head = torch.nn.Linear(hf_model.config.hidden_size, len(cartridge_ids)).to(torch_device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)

    if class_balanced_sampling:
        class_indices: dict[int, list[int]] = {}
        for idx, label in enumerate(train_y.tolist()):
            class_indices.setdefault(label, []).append(idx)

        def sample_batch():
            batch_x = []
            batch_y = []
            for cls_id, indices in class_indices.items():
                sampled = random.sample(indices, min(len(indices), 16))
                batch_x.append(train_x[torch.tensor(sampled)])
                batch_y.extend([cls_id] * len(sampled))
            return torch.cat(batch_x), torch.tensor(batch_y)
    else:
        def sample_batch():
            return train_x, train_y

    best_state = None
    best_val = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        head.train()
        bx, by = sample_batch()
        bx = bx.to(torch_device)
        by = by.to(torch_device)
        optimizer.zero_grad(set_to_none=True)
        logits = head(bx)
        loss = torch.nn.functional.cross_entropy(logits, by)
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            _head = head
            _head.eval()
            with torch.no_grad():
                train_pred = _head(train_x.to(torch_device)).argmax(dim=-1).cpu()
                train_acc = float((train_pred == train_y).float().mean().item())
                val_pred = _head(val_x.to(torch_device)).argmax(dim=-1).cpu()
                val_acc = float((val_pred == val_y).float().mean().item())
            history.append({
                "epoch": epoch, "loss": float(loss.item()),
                "train_accuracy": train_acc, "val_accuracy": val_acc,
            })
            print(f"  epoch={epoch} loss={loss.item():.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}", flush=True)
            if val_acc > best_val:
                best_val = val_acc
                best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}

    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    head.load_state_dict(best_state)

    with torch.no_grad():
        train_pred = head(train_x.to(torch_device)).argmax(dim=-1).cpu()
        train_acc = float((train_pred == train_y).float().mean().item())
        val_pred = head(val_x.to(torch_device)).argmax(dim=-1).cpu()
        val_acc = float((val_pred == val_y).float().mean().item())

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_counts_train = {k: sum(1 for _, l in train_examples if l == v) for k, v in label_by_id.items()}
    class_counts_val = {k: sum(1 for _, l in val_examples if l == v) for k, v in label_by_id.items()}

    confusion = [[0] * len(cartridge_ids) for _ in range(len(cartridge_ids))]
    for true_idx, pred_idx in zip(val_y.tolist(), val_pred.tolist()):
        confusion[true_idx][pred_idx] += 1

    payload = {
        "router_type": "qwen_embedding_linear_v1",
        "model_name": model_name,
        "d_model": int(hf_model.config.hidden_size),
        "cartridge_ids": cartridge_ids,
        "head_state": best_state,
        "confidence_threshold": 0.0,
        "ambiguous_margin": 0.0,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "train_count": len(train_examples),
        "val_count": len(val_examples),
        "class_counts_train": class_counts_train,
        "class_counts_val": class_counts_val,
        "confusion_matrix": confusion,
        "history": history,
    }
    artifact = output_dir / "qwen_learned_router.pt"
    torch.save(payload, artifact)

    metadata = {
        "router_type": "qwen_embedding_linear_v1",
        "model_name": model_name,
        "cartridge_ids": list(cartridge_ids),
        "class_counts_train": class_counts_train,
        "class_counts_val": class_counts_val,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "confusion_matrix": confusion,
    }
    (output_dir / "router_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "train_metrics.jsonl").write_text(
        "\n".join(json.dumps(h) for h in history) + "\n", encoding="utf-8"
    )

    report = {key: value for key, value in payload.items() if key != "head_state"}
    report["artifact"] = str(artifact)
    (output_dir / "qwen_learned_router_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(f"\nRouter trained: val_acc={val_acc:.4f}, artifact={artifact}", flush=True)
    return report
