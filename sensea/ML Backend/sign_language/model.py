import torch
import torch.nn as nn


class SignLanguageModel(nn.Module):
    def __init__(self, input_size=258, num_classes=20):
        super().__init__()

        self.normalization = nn.LayerNorm(input_size)

        self.conv = nn.Sequential(
            nn.Conv1d(
                input_size,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.gru = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, frames, features)
        x = self.normalization(x)

        # Conv1D expects: (batch, channels, frames)
        x = x.transpose(1, 2)

        x = self.conv(x)

        # Back to: (batch, frames, channels)
        x = x.transpose(1, 2)

        # Temporal processing
        x, _ = self.gru(x)

        # Attention over time
        scores = self.attention(x)
        weights = torch.softmax(scores, dim=1)

        x = torch.sum(weights * x, dim=1)

        # Classification
        return self.classifier(x)
