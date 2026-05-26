"""Qwen cartridge runner and trainer for owned cartridge research loops."""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from hybrid.cartridge_harness.core import (
    TaskExample,
    build_summary,
    compare_rows,
    evaluate_text_runner,
)
from hybrid.cartridges import CartridgeManifest, CartridgeRole, SteererCartridgeRack
from hybrid.superposition_steerer_v3 import FeatureConditionedAdapterSteerer


@dataclass
class LoadedQwenCartridge:
    """A deployable Qwen cartridge loaded from a saved harness artifact."""

    path: Path
    manifest: CartridgeManifest
    steerer: FeatureConditionedAdapterSteerer
    history: list[dict]
    summary: dict


@dataclass(frozen=True)
class CartridgeRouteDecision:
    """Control-plane decision from the main router cartridge."""

    cartridge_ids: tuple[str, ...]
    reason: str


def build_qwen_adapter_steerer_from_checkpoint(
    checkpoint: dict,
    *,
    d_model: int,
    device,
) -> FeatureConditionedAdapterSteerer:
    """Reconstruct a Qwen adapter steerer from a `cartridge_best.pt` payload."""

    manifest = checkpoint.get("manifest") or {}
    if manifest.get("steerer_class") != "FeatureConditionedAdapterSteerer":
        raise ValueError(f"unsupported Qwen cartridge steerer: {manifest.get('steerer_class')!r}")
    state = checkpoint["steerer_state"]
    inject_layers = [int(layer) for layer in manifest.get("inject_layers", ())]
    if not inject_layers:
        raise ValueError("Qwen cartridge checkpoint is missing inject_layers")
    bottleneck = _infer_adapter_bottleneck(state)
    steerer = FeatureConditionedAdapterSteerer(
        d_model=d_model,
        inject_layers=inject_layers,
        bottleneck=bottleneck,
        init_scale=0.005,
        noise_scale=0.0,
    ).to(device)
    steerer.load_state_dict(state)
    steerer.eval()
    return steerer


def load_qwen_adapter_cartridge(
    path: str | Path,
    *,
    d_model: int,
    device,
) -> LoadedQwenCartridge:
    """Load one saved Qwen cartridge as an individually mountable unit."""

    import torch

    artifact = Path(path)
    checkpoint = torch.load(artifact, map_location=device, weights_only=False)
    manifest_payload = checkpoint.get("manifest") or {}
    manifest = CartridgeManifest(**manifest_payload)
    steerer = build_qwen_adapter_steerer_from_checkpoint(
        checkpoint,
        d_model=d_model,
        device=device,
    )
    return LoadedQwenCartridge(
        path=artifact,
        manifest=manifest,
        steerer=steerer,
        history=list(checkpoint.get("history", [])),
        summary=dict(checkpoint.get("summary", {})),
    )


class QwenCartridgeRuntime:
    """Frozen Qwen runtime that can mount saved cartridges individually or together."""

    def __init__(self, model_name: str, device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(self.device)
        self.hf_model.eval()
        for param in self.hf_model.parameters():
            param.requires_grad = False
        self.d_model = self.hf_model.config.hidden_size
        self.rack = SteererCartridgeRack()
        self.loaded: dict[str, LoadedQwenCartridge] = {}
        self.prompt_router = QwenMainCartridgeRouter()

    def load_prompt_router(self, path: str | Path):
        self.prompt_router = QwenLearnedCartridgeRouter(path, device=self.device)

    def load_cartridge(self, path: str | Path, *, weight: float = 1.0, active: bool = True) -> LoadedQwenCartridge:
        loaded = load_qwen_adapter_cartridge(path, d_model=self.d_model, device=self.device)
        self.rack.mount(loaded.manifest, loaded.steerer, weight=weight, active=active)
        self.loaded[loaded.manifest.cartridge_id] = loaded
        self.rack.register_hooks(self.hf_model.model)
        return loaded

    def activate_only(self, cartridge_id: str | None):
        for loaded_id in self.loaded:
            self.rack.activate(loaded_id, cartridge_id is not None and loaded_id == cartridge_id)

    def set_all_active(self, active: bool = True):
        for loaded_id in self.loaded:
            self.rack.activate(loaded_id, active)

    def activate_selected(self, cartridge_ids: Iterable[str]):
        selected = set(cartridge_ids)
        for loaded_id in self.loaded:
            self.rack.activate(loaded_id, loaded_id in selected)

    def set_composition_mode(self, mode: str):
        self.rack.set_composition_mode(mode)

    def set_zero_weights(self, seq_len: int, batch_size: int = 1):
        weights = self.torch.zeros(batch_size, seq_len, 21, device=self.device)
        self.rack.set_weights(weights)

    def generate(self, prompt: str, max_tokens: int = 24) -> str:
        ids = list(self.tokenizer.encode(prompt))
        generated: list[int] = []
        with self.torch.no_grad():
            for _ in range(max_tokens):
                inp = self.torch.tensor([ids[-512:]], device=self.device)
                if self.rack.list_active():
                    self.set_zero_weights(inp.shape[1])
                logits = self.hf_model(inp).logits[0, -1].float()
                if not self.torch.isfinite(logits).all():
                    return "<NONFINITE>"
                next_id = int(logits.argmax())
                ids.append(next_id)
                generated.append(next_id)
                if next_id == self.tokenizer.eos_token_id:
                    break
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def prompt_embedding(self, prompt: str):
        active_ids = self.rack.list_active()
        self.activate_only(None)
        ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.hf_model(ids, output_hidden_states=True, use_cache=False)
            hidden = out.hidden_states[-1][0]
            embedding = hidden.mean(dim=0).float()
        self.activate_selected(active_ids)
        return embedding

    def _route_decision(self, prompt: str) -> CartridgeRouteDecision:
        return self.prompt_router.select(prompt, self.loaded.keys(), runtime=self)

    def route_prompt(self, prompt: str) -> str | None:
        decision = self._route_decision(prompt)
        return decision.cartridge_ids[0] if decision.cartridge_ids else None

    def route_prompt_chain(self, prompt: str) -> CartridgeRouteDecision:
        return self._route_decision(prompt)

    def generate_routed(self, prompt: str, max_tokens: int = 24) -> str:
        cartridge_id = self.route_prompt(prompt)
        self.activate_only(cartridge_id)
        return self.generate(prompt, max_tokens=max_tokens)

    def generate_gated_chain(self, prompt: str, max_tokens: int = 24) -> str:
        self.set_composition_mode("chain")
        decision = self.route_prompt_chain(prompt)
        self.activate_selected(decision.cartridge_ids)
        return self.generate(prompt, max_tokens=max_tokens)

    def cleanup(self):
        self.rack.remove_hooks()


def _infer_adapter_bottleneck(state: dict) -> int:
    for key, value in state.items():
        if key.startswith("down.") and key.endswith(".weight"):
            return int(value.shape[0])
    raise ValueError("could not infer adapter bottleneck from checkpoint state")


class QwenMainCartridgeRouter:
    """Main control cartridge deciding which capability cartridges to use.

    This is the safe-composition layer: it keeps capability cartridges mounted,
    selects the relevant subset for a prompt, and lets the rack apply only that
    subset in chain mode. The current implementation is deterministic and
    manifest-style; it can later be replaced by a trained router cartridge with
    the same `select(prompt, available)` ABI.
    """

    _routes = (
        (
            "qwen-private-facts-cartridge",
            (
                "aurora", "private registry", "internal registry", "registry code",
                "access code", "access value", "project aster", "project brindle",
                "project cobalt", "project drift", "project ember", "project fjord",
                "project glade", "project helix", "project iris", "project juno",
                "project kestrel", "project lattice",
            ),
        ),
        (
            "qwen-arithmetic-router-cartridge",
            (
                "arithmetic", "numeric result", "answer number", "solve this arithmetic",
                "give only the numeric", "return only the answer number",
            ),
        ),
        (
            "qwen-code-router-cartridge",
            (
                "code routing", "code cartridge", "programming request", "route label",
                "python", "sql", "javascript", "bash", "function", "query selecting",
            ),
        ),
        (
            "qwen-safety-router-cartridge",
            (
                "safety", "policy routing", "phishing", "password", "malware",
                "account", "security awareness", "password manager",
            ),
        ),
        (
            "qwen-instruction-format-cartridge",
            (
                "instruction-format", "response shape", "response format", "format label",
                "format route", "numbered", "polite email", "one sentence", "table", "brainstorm",
                "explain gravity", "plain_explanation",
            ),
        ),
    )

    def select(self, prompt: str, available: Iterable[str], runtime: QwenCartridgeRuntime | None = None) -> CartridgeRouteDecision:
        available_set = set(available)
        text = prompt.lower()
        scores: list[tuple[int, str]] = []
        for cartridge_id, needles in self._routes:
            if cartridge_id not in available_set:
                continue
            score = sum(1 for needle in needles if needle in text)
            if cartridge_id == "qwen-arithmetic-router-cartridge" and re.search(r"\b\d+\s*(?:\+|-|\*|/)\s*\d+\b", text):
                score += 3
            if score:
                scores.append((score, cartridge_id))
        if not scores:
            return CartridgeRouteDecision((), "no matching cartridge rule")
        scores.sort(reverse=True)
        score, cartridge_id = scores[0]
        return CartridgeRouteDecision((cartridge_id,), f"matched {score} route features")

    def route(self, prompt: str, available: Iterable[str], runtime: QwenCartridgeRuntime | None = None) -> str | None:
        decision = self.select(prompt, available, runtime=runtime)
        return decision.cartridge_ids[0] if decision.cartridge_ids else None


QwenPromptRouter = QwenMainCartridgeRouter


class QwenLearnedCartridgeRouter:
    """Learned main-router cartridge backed by frozen-Qwen prompt embeddings."""

    def __init__(self, path: str | Path, *, device):
        import torch

        self.torch = torch
        self.path = Path(path)
        payload = torch.load(self.path, map_location=device, weights_only=False)
        if payload.get("router_type") != "qwen_embedding_linear_v1":
            raise ValueError(f"unsupported router artifact: {payload.get('router_type')!r}")
        self.cartridge_ids = tuple(payload["cartridge_ids"])
        self.confidence_threshold = float(payload.get("confidence_threshold", 0.0))
        self.ambiguous_margin = float(payload.get("ambiguous_margin", 0.0))
        self.head = torch.nn.Linear(int(payload["d_model"]), len(self.cartridge_ids)).to(device)
        self.head.load_state_dict(payload["head_state"])
        self.head.eval()

    def select(self, prompt: str, available: Iterable[str], runtime: QwenCartridgeRuntime | None = None) -> CartridgeRouteDecision:
        if runtime is None:
            raise ValueError("learned Qwen router requires a runtime for prompt embeddings")
        available_set = set(available)
        embedding = runtime.prompt_embedding(prompt).to(runtime.device)
        with self.torch.no_grad():
            logits = self.head(embedding.unsqueeze(0))[0].float()
            mask = self.torch.tensor(
                [cartridge_id in available_set for cartridge_id in self.cartridge_ids],
                device=logits.device,
                dtype=self.torch.bool,
            )
            logits = logits.masked_fill(~mask, -1e9)
            probs = self.torch.softmax(logits, dim=-1)
            values, indices = self.torch.topk(probs, k=min(2, probs.numel()))
        confidence = float(values[0].item())
        cartridge_id = self.cartridge_ids[int(indices[0].item())]
        if not available_set:
            return CartridgeRouteDecision((), "no mounted cartridges")
        if confidence < self.confidence_threshold:
            return CartridgeRouteDecision((), f"router confidence {confidence:.3f} below threshold")
        if values.numel() > 1:
            margin = float((values[0] - values[1]).item())
            if margin < self.ambiguous_margin:
                return CartridgeRouteDecision((), f"router margin {margin:.3f} below threshold")
        return CartridgeRouteDecision((cartridge_id,), f"learned router confidence {confidence:.3f}")

    def route(self, prompt: str, available: Iterable[str], runtime: QwenCartridgeRuntime | None = None) -> str | None:
        decision = self.select(prompt, available, runtime=runtime)
        return decision.cartridge_ids[0] if decision.cartridge_ids else None


def train_qwen_embedding_router(
    *,
    model_name: str,
    device: str,
    out_dir: str | Path,
    epochs: int = 300,
    lr: float = 3e-3,
    confidence_threshold: float = 0.0,
    ambiguous_margin: float = 0.0,
) -> dict:
    """Train a learned cartridge router from frozen-Qwen prompt embeddings."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from hybrid.cartridge_harness.suites import build_all_suites

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
    cartridge_ids = tuple(suite.cartridge_id for suite in suites)
    label_by_id = {cartridge_id: idx for idx, cartridge_id in enumerate(cartridge_ids)}
    train_examples: list[tuple[str, int]] = []
    val_examples: list[tuple[str, int]] = []
    for suite in suites:
        label = label_by_id[suite.cartridge_id]
        for task in suite.tasks:
            target = train_examples if task.split == "train" else val_examples
            target.append((task.prompt, label))

    def encode(prompt: str):
        ids = tokenizer.encode(prompt, return_tensors="pt").to(torch_device)
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
    train_x_device = train_x.to(torch_device)
    train_y_device = train_y.to(torch_device)
    best_state = None
    best_val = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        logits = head(train_x_device)
        loss = torch.nn.functional.cross_entropy(logits, train_y_device)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            train_acc = _router_accuracy(head, train_x, train_y, torch_device)
            val_acc = _router_accuracy(head, val_x, val_y, torch_device)
            history.append({"epoch": epoch, "loss": float(loss.item()), "train_accuracy": train_acc, "val_accuracy": val_acc})
            if val_acc > best_val:
                best_val = val_acc
                best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}

    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    head.load_state_dict(best_state)
    train_acc = _router_accuracy(head, train_x, train_y, torch_device)
    val_acc = _router_accuracy(head, val_x, val_y, torch_device)
    payload = {
        "router_type": "qwen_embedding_linear_v1",
        "model_name": model_name,
        "d_model": int(hf_model.config.hidden_size),
        "cartridge_ids": cartridge_ids,
        "head_state": best_state,
        "confidence_threshold": float(confidence_threshold),
        "ambiguous_margin": float(ambiguous_margin),
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "train_count": len(train_examples),
        "val_count": len(val_examples),
        "history": history,
    }
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / "qwen_learned_router.pt"
    torch.save(payload, artifact)
    report = {key: value for key, value in payload.items() if key != "head_state"}
    report["artifact"] = str(artifact)
    (output_dir / "qwen_learned_router_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _router_accuracy(head, embeddings, labels, device) -> float:
    import torch

    head.eval()
    with torch.no_grad():
        pred = head(embeddings.to(device)).argmax(dim=-1).cpu()
    return float((pred == labels).float().mean().item())


class QwenBakedLoraRunner:
    """Qwen plus a trainable LoRA adapter that bakes suite behavior into the model."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        *,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(self.device)
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
            bias="none",
        )
        self.model = get_peft_model(base, config)
        self.model.train()
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

    def generate(self, prompt: str, max_tokens: int = 24) -> str:
        ids = list(self.tokenizer.encode(prompt))
        generated: list[int] = []
        self.model.eval()
        with self.torch.no_grad():
            for _ in range(max_tokens):
                inp = self.torch.tensor([ids[-512:]], device=self.device)
                logits = self.model(inp).logits[0, -1].float()
                if not self.torch.isfinite(logits).all():
                    return "<NONFINITE>"
                next_id = int(logits.argmax())
                ids.append(next_id)
                generated.append(next_id)
                if next_id == self.tokenizer.eos_token_id:
                    break
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def save_adapter(self, out_dir: Path):
        self.model.save_pretrained(out_dir / "adapter")
        self.tokenizer.save_pretrained(out_dir / "adapter")


def train_qwen_baked_lora(
    *,
    model_name: str,
    device: str,
    out_dir: str | Path,
    steps: int = 600,
    eval_every: int = 100,
    lr: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    seed: int = 23,
) -> dict:
    """Train a LoRA adapter that bakes router/cartridge suite behavior into Qwen."""

    import torch.nn.functional as F

    from hybrid.cartridge_harness.suites import build_all_suites

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = QwenBakedLoraRunner(
        model_name,
        device,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    torch = runner.torch
    random.seed(seed)
    torch.manual_seed(seed)
    suites = build_all_suites()
    tasks = [task for suite in suites for task in suite.tasks]
    train_tasks = [task for task in tasks if task.split == "train"]
    eval_tasks = tasks
    optimizer = torch.optim.AdamW(
        [param for param in runner.model.parameters() if param.requires_grad],
        lr=lr,
        weight_decay=0.0,
    )
    history: list[dict] = []
    best_key = (-1, -1)
    best_adapter_dir = output_dir / "best_adapter"

    for step in range(1, steps + 1):
        runner.model.train()
        row = random.choice(train_tasks)
        prompt_ids = runner.tokenizer.encode(row.prompt)
        full_ids = runner.tokenizer.encode(f"{row.prompt} {row.expected}\n")
        x = torch.tensor([full_ids[:-1]], device=runner.device)
        y = torch.tensor([full_ids[1:]], device=runner.device)
        mask = torch.zeros_like(y, dtype=torch.float32)
        mask[:, max(0, len(prompt_ids) - 1):] = 1.0
        logits = runner.model(x).logits.float()
        loss_tokens = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1),
            reduction="none",
        ).reshape_as(mask)
        loss = (loss_tokens * mask).sum() / mask.sum().clamp(min=1.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(runner.model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % eval_every == 0 or step == steps:
            rows = evaluate_text_runner(eval_tasks, runner.generate)
            summary = build_summary(rows)
            split = summary.by_split
            key = (split.get("heldout", {}).get("correct", 0), summary.correct)
            item = {"step": step, "loss": float(loss.detach().cpu()), **summary.to_json()}
            history.append(item)
            print(
                f"[baked-lora] step={step} loss={item['loss']:.4f} "
                f"correct={summary.correct}/{summary.total} heldout="
                f"{split.get('heldout', {}).get('correct', 0)}/"
                f"{split.get('heldout', {}).get('total', 0)}",
                flush=True,
            )
            if key > best_key:
                best_key = key
                runner.model.save_pretrained(best_adapter_dir)
                runner.tokenizer.save_pretrained(best_adapter_dir)
            if summary.correct == summary.total:
                print(f"[baked-lora] early_stop step={step} perfect_eval=1", flush=True)
                break

    runner.model.save_pretrained(output_dir / "adapter_last")
    runner.tokenizer.save_pretrained(output_dir / "adapter_last")
    final_rows = evaluate_text_runner(eval_tasks, runner.generate)
    final_summary = build_summary(final_rows)
    result = {
        "model": model_name,
        "artifact": str(best_adapter_dir),
        "last_adapter": str(output_dir / "adapter_last"),
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "train_count": len(train_tasks),
        "eval_count": len(eval_tasks),
        "history": history,
        "final_summary": final_summary.to_json(),
        "final_rows": [row.to_json() for row in final_rows],
    }
    (output_dir / "baked_lora_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


class QwenAdapterCartridgeRunner:
    """Frozen Qwen plus a trainable feature-conditioned cartridge."""

    def __init__(self, model_name: str, device: str = "cuda", bottleneck: int = 64,
                 cartridge_id: str = "owned-qwen-adapter-cartridge",
                 role: str | CartridgeRole = CartridgeRole.DOMAIN_CAPABILITY,
                 source_corpus: str = "hybrid.cartridge_harness"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(self.device)
        self.hf_model.eval()
        for param in self.hf_model.parameters():
            param.requires_grad = False

        self.d_model = self.hf_model.config.hidden_size
        self.inject_layers = [
            idx for idx in (0, 2, 4, 7, 10, 14, 17, 21, 24, 26)
            if idx < len(self.hf_model.model.layers)
        ]
        self.steerer = FeatureConditionedAdapterSteerer(
            d_model=self.d_model,
            inject_layers=self.inject_layers,
            bottleneck=bottleneck,
            init_scale=0.005,
            noise_scale=0.0,
        ).to(self.device)
        for gamma in self.steerer.gammas.values():
            gamma.data.fill_(0.02)

        self.manifest = CartridgeManifest(
            cartridge_id=cartridge_id,
            role=role,
            base_model_id=model_name,
            tokenizer_id=model_name,
            steerer_class="FeatureConditionedAdapterSteerer",
            inject_layers=tuple(self.inject_layers),
            parameter_count=sum(param.numel() for param in self.steerer.parameters()),
            source_corpus=source_corpus,
            metadata={"runtime": "owned-cartridge-harness"},
        )
        self.rack = SteererCartridgeRack()
        self.rack.mount(self.manifest, self.steerer, weight=1.0, active=False)
        self.rack.register_hooks(self.hf_model.model)
        self.enabled = False

    def set_enabled(self, enabled: bool):
        self.enabled = enabled
        self.rack.activate(self.manifest.cartridge_id, enabled)

    def set_zero_weights(self, seq_len: int, batch_size: int = 1):
        weights = self.torch.zeros(batch_size, seq_len, 21, device=self.device)
        self.rack.set_weights(weights)

    def generate(self, prompt: str, max_tokens: int = 24) -> str:
        ids = list(self.tokenizer.encode(prompt))
        generated: list[int] = []
        with self.torch.no_grad():
            for _ in range(max_tokens):
                inp = self.torch.tensor([ids[-512:]], device=self.device)
                if self.enabled:
                    self.set_zero_weights(inp.shape[1])
                else:
                    self.rack.activate(self.manifest.cartridge_id, False)
                logits = self.hf_model(inp).logits[0, -1].float()
                if not self.torch.isfinite(logits).all():
                    return "<NONFINITE>"
                next_id = int(logits.argmax())
                ids.append(next_id)
                generated.append(next_id)
                if next_id == self.tokenizer.eos_token_id:
                    break
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def cleanup(self):
        self.rack.remove_hooks()


def train_answer_cartridge(
    runner: QwenAdapterCartridgeRunner,
    train_tasks: list[TaskExample],
    eval_tasks: list[TaskExample],
    out_dir: Path,
    steps: int = 700,
    eval_every: int = 50,
    lr: float = 6e-4,
    seed: int = 23,
) -> dict:
    """Train only the mounted cartridge on prompt/answer rows."""

    torch = runner.torch
    random.seed(seed)
    torch.manual_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    runner.set_enabled(False)
    baseline_rows = evaluate_text_runner(eval_tasks, runner.generate)
    baseline_summary = build_summary(baseline_rows)
    print(
        f"[baseline] {baseline_summary.correct}/{baseline_summary.total} "
        f"acc={baseline_summary.accuracy:.3f}",
        flush=True,
    )

    runner.set_enabled(True)
    runner.steerer.train()
    optimizer = torch.optim.AdamW(runner.steerer.parameters(), lr=lr, weight_decay=0.01)
    best_key = (-1, -1)
    best_state = None
    history: list[dict] = []
    for step in range(1, steps + 1):
        loss = _train_step(runner, train_tasks, optimizer)
        if step % eval_every == 0:
            runner.steerer.eval()
            rows = evaluate_text_runner(eval_tasks, runner.generate)
            summary = build_summary(rows)
            split = summary.by_split
            key = (split.get("heldout", {}).get("correct", 0), summary.correct)
            history.append({"step": step, "loss": loss, **summary.to_json()})
            print(
                f"[eval] step={step} loss={loss:.4f} "
                f"correct={summary.correct}/{summary.total} "
                f"acc={summary.accuracy:.3f} heldout="
                f"{split.get('heldout', {}).get('correct', 0)}/"
                f"{split.get('heldout', {}).get('total', 0)}",
                flush=True,
            )
            if key > best_key:
                best_key = key
                best_state = {
                    state_key: value.detach().cpu().clone()
                    for state_key, value in runner.steerer.state_dict().items()
                }
                torch.save(
                    {
                        "steerer_state": best_state,
                        "manifest": runner.manifest.__dict__,
                        "history": history,
                        "summary": summary.to_json(),
                    },
                    out_dir / "cartridge_best.pt",
                )
            if summary.correct == summary.total:
                print(f"[early_stop] step={step} perfect_eval=1", flush=True)
                break
            runner.steerer.train()

    runner.steerer.eval()
    if best_state is not None:
        runner.steerer.load_state_dict(best_state, strict=False)
    cartridge_rows = evaluate_text_runner(eval_tasks, runner.generate)
    cartridge_summary = build_summary(cartridge_rows)
    print(
        f"[final] {cartridge_summary.correct}/{cartridge_summary.total} "
        f"acc={cartridge_summary.accuracy:.3f}",
        flush=True,
    )
    comparison = compare_rows(baseline_rows, cartridge_rows)
    result = {
        "model": runner.model_name,
        "artifact": str(out_dir / "cartridge_best.pt"),
        "baseline_summary": baseline_summary.to_json(),
        "cartridge_summary": cartridge_summary.to_json(),
        "history": history,
        "baseline_rows": [row.to_json() for row in baseline_rows],
        "cartridge_rows": [row.to_json() for row in cartridge_rows],
        **comparison,
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _train_step(runner: QwenAdapterCartridgeRunner, tasks: list[TaskExample], optimizer) -> float:
    import torch.nn.functional as F

    row = random.choice(tasks)
    target_text = f"{row.prompt} {row.expected}\n"
    prompt_ids = runner.tokenizer.encode(row.prompt)
    full_ids = runner.tokenizer.encode(target_text)
    x = runner.torch.tensor([full_ids[:-1]], device=runner.device)
    y = runner.torch.tensor([full_ids[1:]], device=runner.device)
    mask = runner.torch.zeros_like(y, dtype=runner.torch.float32)
    mask[:, max(0, len(prompt_ids) - 1):] = 1.0
    runner.set_zero_weights(x.shape[1])
    logits = runner.hf_model(x).logits.float()
    loss_tokens = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        y.reshape(-1),
        reduction="none",
    ).reshape_as(mask)
    loss = (loss_tokens * mask).sum() / mask.sum().clamp(min=1.0)
    loss = loss + 0.00005 * runner.steerer.orthogonal_penalty()
    optimizer.zero_grad()
    loss.backward()
    runner.torch.nn.utils.clip_grad_norm_(runner.steerer.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach().cpu())


def split_tasks(tasks: Iterable[TaskExample]) -> tuple[list[TaskExample], list[TaskExample]]:
    materialized = list(tasks)
    return (
        [task for task in materialized if task.split == "train"],
        materialized,
    )