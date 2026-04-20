import torch

from config import Config
from model import TextCNN
from utils import clean_text, encode_text


def load_model_and_vocab(model_path):
    checkpoint = torch.load(model_path, map_location=Config.DEVICE)

    vocab = checkpoint["vocab"]
    cfg = checkpoint["config"]

    model = TextCNN(
        vocab_size=len(vocab),
        embed_dim=cfg["embed_dim"],
        num_classes=cfg["num_classes"],
        num_filters=cfg["num_filters"],
        filter_sizes=cfg["filter_sizes"],
        dropout=cfg["dropout"],
        padding_idx=vocab[cfg["pad_token"]],
    ).to(Config.DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, vocab


def predict_sentiment(text, model, vocab):
    text = clean_text(text)
    input_ids = encode_text(
        text=text,
        vocab=vocab,
        max_len=Config.MAX_LEN,
        unk_token=Config.UNK_TOKEN,
        pad_token=Config.PAD_TOKEN,
    )

    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(Config.DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    label_map = {0: "negative", 1: "positive"}

    return {
        "label": label_map[pred],
        "probabilities": probs.squeeze(0).cpu().numpy().tolist(),
    }


if __name__ == "__main__":
    model, vocab = load_model_and_vocab(Config.MODEL_SAVE_PATH)

    sample_text = "This movie was surprisingly good, I really enjoyed it."
    result = predict_sentiment(sample_text, model, vocab)

    print("Input:", sample_text)
    print("Prediction:", result["label"])
    print("Probabilities:", result["probabilities"])
