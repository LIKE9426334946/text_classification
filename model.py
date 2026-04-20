import torch
import torch.nn as nn


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_classes,
        num_filters,
        filter_sizes,
        dropout=0.5,
        padding_idx=0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=padding_idx,
        )

        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,
                out_channels=num_filters,
                kernel_size=(fs, embed_dim),
            )
            for fs in filter_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids)     # (B, L, D)
        x = x.unsqueeze(1)                # (B, 1, L, D)

        conv_outputs = []
        for conv in self.convs:
            c = conv(x)                   # (B, F, L-fs+1, 1)
            c = torch.relu(c).squeeze(3)  # (B, F, L-fs+1)
            p = torch.max_pool1d(c, kernel_size=c.size(2)).squeeze(2)  # (B, F)
            conv_outputs.append(p)

        out = torch.cat(conv_outputs, dim=1)  # (B, F * len(filter_sizes))
        out = self.dropout(out)
        logits = self.fc(out)                 # (B, num_classes)
        return logits
