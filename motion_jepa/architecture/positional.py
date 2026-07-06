
import torch as t
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, segment_count: int, group_count: int, embed_dim: int):
        super().__init__()

        self.segment_count = segment_count
        self.group_count = group_count
        self.embed_dim = embed_dim

        # Temporal encoding
        self.encode_t = nn.Parameter(t.zeros(1, self.segment_count, 1, self.embed_dim))
        nn.init.trunc_normal_(self.encode_t, std=0.02)

        # Group encoding
        self.encode_g = nn.Parameter(t.zeros(1, 1, self.group_count, self.embed_dim))
        nn.init.trunc_normal_(self.encode_g,  std=0.02)

    def grid(self) -> t.Tensor:
        return self.encode_t + self.encode_g # 1, S, G, E

    def flat(self) -> t.Tensor:
        return self.grid().flatten(1, 2) # 1, SG, E

    def add_grid(self, x: t.Tensor) -> t.Tensor:
        return x + self.grid() # B, S, G, E

    def add_flat(self, x: t.Tensor, idx: t.Tensor | None = None) -> t.Tensor:
        if idx is not None:
            return x + self.gather(idx)
        return x + self.flat()
    
    def gather(self, idx: t.Tensor) -> t.Tensor:
        idx = idx.to(device=self.encode_t.device, dtype=t.long)
        pos = self.flat().expand(idx.shape[0], -1, -1)
        idx = idx.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        return pos.gather(dim=1, index=idx)