"""Generate text with a trained CompiledFeatureTransformer checkpoint."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from hybrid.compiled_features import (
    CompiledFeatureTransformer,
    CompiledFeatureTransformerConfig,
    GPT2CompiledChannelBuilder,
    GPT2CompiledChannelConfig,
    build_token_stat_features_for_span,
)
from hybrid.decoding import DecodingConfig, deterministic_generate


class CompiledFeatureRuntime:
    """Callable wrapper that rebuilds causal compiled features per decode step."""

    def __init__(
        self,
        model: CompiledFeatureTransformer,
        *,
        history: int,
        window: int,
        compiled_builder: GPT2CompiledChannelBuilder | None = None,
    ):
        self.model = model
        self.history = history
        self.window = window
        self.compiled_builder = compiled_builder

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape (B, T), got {tuple(input_ids.shape)}")
        device = input_ids.device
        rows = []
        for row in input_ids.detach().cpu():
            if self.compiled_builder is None:
                features = build_token_stat_features_for_span(
                    row,
                    start=0,
                    length=row.numel(),
                    history=self.history,
                    window=self.window,
                )
            else:
                features = self.compiled_builder.build_features_for_span(
                    row,
                    start=0,
                    length=row.numel(),
                    history=self.history,
                )
            rows.append(features)
        features = torch.stack(rows, dim=0).to(device)
        return self.model(input_ids, features)


def load_checkpoint(path: Path, device: torch.device) -> tuple[CompiledFeatureTransformer, dict]:
    ckpt = torch.load(path, map_location=device)
    cfg = CompiledFeatureTransformerConfig(**ckpt["config"])
    model = CompiledFeatureTransformer(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default="The")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--history", type=int, default=512)
    parser.add_argument("--feature-window", type=int, default=128)
    parser.add_argument("--feature-source", choices=["auto", "token_stat", "compiled_ngram"], default="auto")
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/wikitext_gpt2"))
    parser.add_argument("--compile-max-train-tokens", type=int, default=0)
    parser.add_argument("--compile-alpha", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    device = torch.device(args.device)
    model, ckpt = load_checkpoint(args.ckpt, device)
    feature_source = args.feature_source
    if feature_source == "auto":
        feature_source = ckpt.get("feature_source", ckpt.get("args", {}).get("feature_source", "token_stat"))

    compiled_builder = None
    if feature_source == "compiled_ngram":
        train_path = args.data_dir / "train_ids.pt"
        if not train_path.exists():
            raise FileNotFoundError(f"Missing {train_path}; needed to rebuild compiled_ngram features")
        train_ids = torch.load(train_path, weights_only=False).long()
        compiled_builder = GPT2CompiledChannelBuilder.from_ids(
            train_ids,
            GPT2CompiledChannelConfig(
                alpha=args.compile_alpha,
                max_train_tokens=args.compile_max_train_tokens,
                recency_window=args.feature_window,
            ),
        )
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        prompt_ids = [tokenizer.eos_token_id]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    runtime = CompiledFeatureRuntime(
        model,
        history=args.history,
        window=args.feature_window,
        compiled_builder=compiled_builder,
    )
    cfg = DecodingConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = deterministic_generate(runtime, input_ids, cfg)
    text = tokenizer.decode(generated[0].tolist())

    print("=" * 72)
    print(" COMPILED-FEATURE TRANSFORMER GENERATION")
    print("=" * 72)
    print(f"checkpoint: {args.ckpt}")
    print(f"feature_source: {feature_source}")
    print(f"epoch: {ckpt.get('epoch', '?')} val_ppl: {ckpt.get('val_report', {}).get('ppl', '?')}")
    print("-" * 72)
    print(text)
    print("=" * 72)


if __name__ == "__main__":
    main()
