import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from config import Config
from utils import clean_text, build_vocab, encode_text


class IMDBDataset(Dataset):
    def __init__(self, dataframe, vocab):
        self.texts = dataframe["review"].tolist()
        self.labels = dataframe["label"].tolist()
        self.vocab = vocab

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        input_ids = encode_text(
            text=text,
            vocab=self.vocab,
            max_len=Config.MAX_LEN,
            unk_token=Config.UNK_TOKEN,
            pad_token=Config.PAD_TOKEN,
        )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }


def load_data_and_build_vocab():
    df = pd.read_csv(Config.CSV_PATH)

    if "review" not in df.columns or "sentiment" not in df.columns:
        raise ValueError("CSV 文件必须包含 'review' 和 'sentiment' 两列")

    df["review"] = df["review"].astype(str).apply(clean_text)
    df["label"] = df["sentiment"].map({"negative": 0, "positive": 1})

    if df["label"].isnull().any():
        raise ValueError("sentiment 列中只能包含 'positive' 或 'negative'")

    df = df.sample(frac=1.0, random_state=Config.SEED).reset_index(drop=True)

    split_idx = int(len(df) * Config.TRAIN_RATIO)
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()

    vocab = build_vocab(
        texts=train_df["review"].tolist(),
        max_vocab_size=Config.MAX_VOCAB_SIZE,
        pad_token=Config.PAD_TOKEN,
        unk_token=Config.UNK_TOKEN,
    )

    return train_df, val_df, vocab


def build_dataloaders():
    train_df, val_df, vocab = load_data_and_build_vocab()

    train_dataset = IMDBDataset(train_df, vocab)
    val_dataset = IMDBDataset(val_df, vocab)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader, vocab
