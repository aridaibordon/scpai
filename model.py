import torch.nn as nn


class SCPAI_H(nn.Module):
    def __init__(self, egrid: list) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(len(egrid), 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )

    def forward(self, x):
        return self.model(x)
