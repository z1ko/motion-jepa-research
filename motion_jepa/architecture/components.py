
import torch as t
import torch.nn as nn

from omegaconf import DictConfig

class TokenizeGroups(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()

        # Store group indices
        self.groups = config.training.groups
        for i, group in enumerate(self.groups):
            self.register_buffer(
                f"group_indices_{i}", 
                t.tensor(self.groups[group], dtype=t.long)
            )

    def forward(self, x: t.Tensor) -> dict[str, t.Tensor]:
        """
        x: tensor of shape ..., D, C
        returns:
            dictionary of groups, each one is a tensor of shape ..., G(i), C
        """

        *_, D, C = x.shape

        groups: dict[str, t.Tensor] = {}
        for i, group in enumerate(self.groups.keys()):

            index: t.Tensor = getattr(self, f"group_indices_{i}")
            if index.max() >= D or index.min() < 0:
                raise ValueError(f"Group {group!r} has indices outside input D={D}")
            
            x_group = x.index_select(dim=-2, index=index)
            groups[group] = x_group

        return groups
    
class TokenizeSegments(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.segment_size = config.architecture.segment_size

    def forward(self, x: t.Tensor) -> t.Tensor:

        B, T, D, C = x.shape
        if T % self.segment_size != 0:
            raise ValueError(
                f"Input tensor of shape {x.shape} is not compatible with segment_size={self.segment_size}"
            )
        
        segment_count = T // self.segment_size
        x = x.reshape(B, segment_count, self.segment_size, D, C)
        return x
    
class TokenEmbed(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.embed_dim = config.architecture.embed_dim
        self.segment_size = config.architecture.segment_size
        self.channels = config.architecture.channels
        self.groups = config.training.groups

        # Tokenizers
        self.tokenize_t = TokenizeSegments(config)
        self.tokenize_g = TokenizeGroups(config)

        # Common embedding dimension projections
        self.projections = nn.ModuleDict({
            group: nn.Linear(
                self.segment_size * len(indices) * self.channels,
                self.embed_dim
            ) for group, indices in self.groups.items()
        })

    def forward(self, x: t.Tensor) -> t.Tensor:
        
        # Tokenize input motion
        groups = self.tokenize_g(self.tokenize_t(x))

        # Embed to common dimension
        results: list[t.Tensor] = []
        for group in self.groups.keys():
            x_group = groups[group]

            B, S, T, G, C = x_group.shape
            x_group = x_group.reshape(B, S, -1) # B, S, TGC
            x_group = self.projections[group](x_group) # B, S, E
            results.append(x_group)

        return t.stack(results, dim=2) # B, S, G, E

# Standard transformer encoder
def _encoder(d_model: int, depth: int, heads: int, mlp_ratio: float, dropout: float) -> nn.TransformerEncoder:
    return nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        ),
        num_layers=depth
    )