import re
import random
from collections import Counter

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_text(text: str) -> str:
    text = re.sub(r"<.*?>", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s.,!?']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    return re.findall(r"\w+|[^\w\s]", text)


def build_vocab(texts, max_vocab_size, pad_token, unk_token):
    counter = Counter()

    for text in texts:
        tokens = tokenize(text)
        counter.update(tokens)

    most_common = counter.most_common(max_vocab_size - 2)

    vocab = {
        pad_token: 0,
        unk_token: 1,
    }

    for word, _ in most_common:
        vocab[word] = len(vocab)

    return vocab


def encode_text(text, vocab, max_len, unk_token, pad_token):
    tokens = tokenize(text)
    token_ids = [vocab.get(tok, vocab[unk_token]) for tok in tokens]

    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len]
    else:
        token_ids += [vocab[pad_token]] * (max_len - len(token_ids))

    return token_ids
