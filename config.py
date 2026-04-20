import torch

class Config:
    SEED = 42

    CSV_PATH = "/kaggle/input/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv"
    MODEL_SAVE_PATH = "best_textcnn.pt"

    MAX_VOCAB_SIZE = 30000
    MAX_LEN = 200

    BATCH_SIZE = 64
    EMBED_DIM = 128
    NUM_FILTERS = 100
    FILTER_SIZES = [3, 4, 5]
    DROPOUT = 0.5
    NUM_CLASSES = 2

    LR = 1e-3
    EPOCHS = 10
    TRAIN_RATIO = 0.8

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
