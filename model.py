import torch.nn as nn


class SCPAI_H(nn.Module):
    def __init__(self, egrid: list) -> None:
        super().__init__()

        self.name = "SCPAI_H"
        self.description = "SCPAI for homogeneous plasmas"

        self.model = nn.Sequential(
            nn.Linear(len(egrid), 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        return self.model(x)
    

class SCPAI_H2(nn.Module):
    def __init__(self, egrid: list) -> None:
        super().__init__()

        self.name = "SCPAI_H"
        self.description = "SCPAI for homogeneous plasmas"

        self.model = nn.Sequential(
            nn.Linear(len(egrid), 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        return self.model(x)



class SCPAI_MZ(nn.Module):
    def __init__(self, egrid: list, nzones: int) -> None:
        super().__init__()

        self.name = "SCPAI_MZ"
        self.description = f"SCPAI for multizone analysis ({nzones} zones)"

        self.model = nn.Sequential(
            nn.Linear(len(egrid), 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3 * nzones),
        )

    def forward(self, x):
        return self.model(x)
