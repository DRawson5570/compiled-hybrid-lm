"""hf_deepseek.py — Register DeepCausalLM as a HuggingFace PreTrainedModel.

Uses nn.TransformerEncoderLayer (nn.Linear internally) so bitsandbytes
can quantize the FFN and attention projection layers automatically.
GPT-2 BPE tokenizer (V=50257) — compiled priors match.
"""
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


class DeepSeekConfig(PretrainedConfig):
    model_type = "deepseek_lm"

    def __init__(self, vocab_size=50257, d_model=768, n_layers=12, n_heads=12,
                 d_ff=3072, max_len=512, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_len = max_len
        self.dropout = dropout


class DeepSeekPreTrainedModel(PreTrainedModel):
    config_class = DeepSeekConfig
    base_model_prefix = "deepseek_lm"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class DecoderLayer(nn.Module):
    """Single transformer decoder layer with explicit Linear layers.

    Uses separate Q/K/V/O projections (not fused MultiheadAttention) so
    ZeroQ and bitsandbytes can quantize each Linear independently.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn1 = nn.Linear(d_model, d_ff)
        self.ffn2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        h = self.n_heads

        # Self-attention
        residual = x
        x = self.norm1(x)
        q = self.q_proj(x).view(B, T, h, d // h).transpose(1, 2)
        k = self.k_proj(x).view(B, T, h, d // h).transpose(1, 2)
        v = self.v_proj(x).view(B, T, h, d // h).transpose(1, 2)

        scale = (d // h) ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = attn + causal_mask
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, d)
        out = self.o_proj(out)
        x = residual + self.dropout(out)

        # FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn2(F.gelu(self.ffn1(x)))
        x = residual + self.dropout(x)

        return x


class DeepSeekForCausalLM(DeepSeekPreTrainedModel):
    """HF-compatible decoder transformer with explicit Linear attention layers.

    All Linear layers are individual nn.Linear — bitsandbytes can quantize
    each independently. ZeroQ sharding works at the parameter level.
    """

    def __init__(self, config: DeepSeekConfig, **kwargs):
        super().__init__(config)
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            DecoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head_bias = nn.Parameter(torch.zeros(config.vocab_size))

        self.post_init()

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        positions = positions.clamp(0, self.config.max_len - 1)

        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        x = self.dropout(x)

        causal_mask = torch.triu(
            torch.full((T, T), float('-inf'), device=x.device), diagonal=1)

        for layer in self.layers:
            x = layer(x, causal_mask)

        x = self.ln_f(x)

        logits = torch.matmul(x, self.tok_emb.weight.T) + self.head_bias

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, self.config.vocab_size),
                labels[:, 1:].reshape(-1),
            )

        return CausalLMOutputWithCrossAttentions(loss=loss, logits=logits)

    def gradient_checkpointing_enable(self, **kwargs):
        pass


from transformers import AutoConfig
AutoConfig.register("deepseek_lm", DeepSeekConfig)
