"""Compiled-feature hybrid model interfaces."""

from .feature_transformer import CompiledFeatureTransformer, CompiledFeatureTransformerConfig
from .gpt2_compiled_channels import (
    GPT2_COMPILED_CHANNEL_NAMES,
    GPT2_COMPILED_FEATURE_DIM,
    GPT2CompiledChannelBuilder,
    GPT2CompiledChannelConfig,
)
from .gpt2_feature_adapter import (
    CompiledFeatureBatch,
    build_token_stat_features,
    build_token_stat_features_for_span,
    iter_compiled_feature_batches,
    iter_span_compiled_feature_batches,
)

__all__ = [
    "CompiledFeatureTransformer",
    "CompiledFeatureTransformerConfig",
    "GPT2_COMPILED_CHANNEL_NAMES",
    "GPT2_COMPILED_FEATURE_DIM",
    "GPT2CompiledChannelBuilder",
    "GPT2CompiledChannelConfig",
    "CompiledFeatureBatch",
    "build_token_stat_features",
    "build_token_stat_features_for_span",
    "iter_compiled_feature_batches",
    "iter_span_compiled_feature_batches",
]
