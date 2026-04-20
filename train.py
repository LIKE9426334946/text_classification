import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from dataset import build_dataloaders
from model import TextCNN
from utils import set_seed


def evaluate(model, data_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)

            total_loss += loss.item() * input_ids.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_count += input_ids.size(0)

    avg_loss = total_loss / total_count
    avg_acc = total_correct / total_count
    return avg_loss, avg_acc


def train():
    set_seed(Config.SEED)

    train_loader, val_loader, vocab = build_dataloaders()

    model = TextCNN(
        vocab_size=len(vocab),
        embed_dim=Config.EMBED_DIM,
        num_classes=Config.NUM_CLASSES,
        num_filters=Config.NUM_FILTERS,
        filter_sizes=Config.FILTER_SIZES,
        dropout=Config.DROPOUT,
        padding_idx=vocab[Config.PAD_TOKEN],
    ).to(Config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)

    best_val_acc = 0.0

    checkpoint = {
        "vocab": vocab,
        "config": {
            "embed_dim": Config.EMBED_DIM,
            "num_classes": Config.NUM_CLASSES,
            "num_filters": Config.NUM_FILTERS,
            "filter_sizes": Config.FILTER_SIZES,
            "dropout": Config.DROPOUT,
            "pad_token": Config.PAD_TOKEN,
        }
    }

    for epoch in range(Config.EPOCHS):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(Config.DEVICE)
            labels = batch["label"].to(Config.DEVICE)

            optimizer.zero_grad()

            logits = model(input_ids)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * input_ids.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_count += input_ids.size(0)

        train_loss = total_loss / total_count
        train_acc = total_correct / total_count

        val_loss, val_acc = evaluate(model, val_loader, criterion, Config.DEVICE)

        print(
            f"Epoch [{epoch + 1}/{Config.EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint["model_state_dict"] = model.state_dict()
            torch.save(checkpoint, Config.MODEL_SAVE_PATH)
            print(f"Saved best model to: {Config.MODEL_SAVE_PATH}")

    print(f"Training finished. Best Val Acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    train()
