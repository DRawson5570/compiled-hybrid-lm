"""NER features — per-token entity type channels from spaCy.

Provides 5 binary channels per GPT-2 token:
  0: is_person (PER, PERSON)
  1: is_location (LOC, GPE, FAC)
  2: is_organization (ORG)
  3: is_date_or_number (DATE, TIME, MONEY, CARDINAL, QUANTITY, ORDINAL, PERCENT)
  4: is_other_entity (any other NER label: PRODUCT, EVENT, LAW, LANGUAGE, etc.)

These features are precomputed per input sequence and concatenated to the
21 statistical channels to create a 26-channel feature vector for the
cartridge.
"""
from __future__ import annotations

import torch


ENTITY_MAP: dict[str, int] = {
    'PERSON': 0, 'PER': 0,
    'LOC': 1, 'GPE': 1, 'FAC': 1,
    'ORG': 2,
    'DATE': 3, 'TIME': 3, 'MONEY': 3, 'CARDINAL': 3,
    'QUANTITY': 3, 'ORDINAL': 3, 'PERCENT': 3,
}

NER_DIM = 5


def compute_ner_features(text: str, nlp) -> torch.Tensor:
    """Run spaCy NER on raw text and produce per-character entity label."""
    doc = nlp(text)
    char_labels = torch.zeros(len(text), NER_DIM)
    for ent in doc.ents:
        channel = ENTITY_MAP.get(ent.label_, 4)
        char_labels[ent.start_char:ent.end_char, channel] = 1.0
    return char_labels


def align_ner_to_tokens(text: str, offset_mapping: list[tuple[int, int]],
                        char_labels: torch.Tensor) -> torch.Tensor:
    """Map per-character NER labels to per-token features using offset mapping.

    For each token, if ANY of its characters fall inside an entity span, the
    token receives that entity label.  Tokens with no entity chars get zeros.
    """
    token_features = torch.zeros(len(offset_mapping), NER_DIM)
    for idx, (start, end) in enumerate(offset_mapping):
        if start >= end or start >= len(text):
            continue
        token_labels = char_labels[start:min(end, len(text))]
        if token_labels.numel() > 0:
            token_features[idx] = (token_labels.sum(dim=0) > 0).float()
    return token_features


def get_ner_features_for_ids(token_ids: list[int], tokenizer, nlp,
                              pad_to: int | None = None) -> torch.Tensor:
    """Compute 5-channel NER features for a sequence of GPT-2 token IDs.

    Returns (T, 5) float tensor where T is the number of tokens.
    """
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offset_mapping = encoding['offset_mapping']

    char_labels = compute_ner_features(text, nlp)
    token_features = align_ner_to_tokens(text, offset_mapping, char_labels)

    if pad_to is not None and token_features.shape[0] < pad_to:
        pad = torch.zeros(pad_to - token_features.shape[0], NER_DIM)
        token_features = torch.cat([token_features, pad], dim=0)
    elif pad_to is not None:
        token_features = token_features[:pad_to]

    return token_features


def precompute_ner_batch(texts: list[str], tokenizer, nlp) -> list[torch.Tensor]:
    """Precompute NER features for a batch of texts.  Returns list of (T_i, 5) tensors."""
    docs = list(nlp.pipe(texts, batch_size=64))
    results = []
    for text, doc in zip(texts, docs):
        char_labels = torch.zeros(len(text), NER_DIM)
        for ent in doc.ents:
            channel = ENTITY_MAP.get(ent.label_, 4)
            char_labels[ent.start_char:ent.end_char, channel] = 1.0
        encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        offset_mapping = encoding['offset_mapping']
        tf = align_ner_to_tokens(text, offset_mapping, char_labels)
        results.append(tf)
    return results
