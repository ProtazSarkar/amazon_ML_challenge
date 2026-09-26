import torch
from torch import nn


class BranchedEntityMatcher(nn.Module):
    """Name/address branches with country similarity fused at the output stage."""

    def __init__(self, name_features=5, address_features=5, branch_width=8, fusion_width=8):
        super().__init__()
        self.name_branch = nn.Sequential(
            nn.Linear(name_features, branch_width),
            nn.ReLU(),
        )
        self.address_branch = nn.Sequential(
            nn.Linear(address_features, branch_width),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(branch_width + branch_width + 1, fusion_width),
            nn.ReLU(),
            nn.Linear(fusion_width, 1),
        )

    def forward(self, name_features, address_features, country_similarity):
        name_embedding = self.name_branch(name_features)
        address_embedding = self.address_branch(address_features)
        fused_features = torch.cat(
            [name_embedding, address_embedding, country_similarity], dim=1
        )
        return self.fusion(fused_features).squeeze(1)
