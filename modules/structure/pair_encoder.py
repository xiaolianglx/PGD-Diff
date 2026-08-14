# modules/structure/pair_encoder.py
import torch
import torch.nn as nn

class PairEncoder(nn.Module):
    """
    轻量级 Pair 编码器：从残基坐标计算 Pair Bias
    输入：坐标，形状为 (total_residues, 4, 3) 或 (total_residues, 3)
    输出：pair_bias 矩阵列表，每个元素形状 (L_i, L_i)
    """
    def __init__(self, hidden_dim=32, output_dim=1, use_mlp=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_mlp = use_mlp

        if use_mlp:
            self.distance_mlp = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            self.register_buffer('scale', torch.tensor(1.0))

    def forward(self, pos, batch_index=None):
        # ---------- 1. 提取 CA 原子坐标，确保输出 (total_residues, 3) ----------
        if pos.dim() == 2 and pos.size(-1) == 3:
            ca = pos
        elif pos.dim() == 3:
            # 形状可能是 (total_residues, 4, 3) 或 (batch, max_len, 3)
            if pos.size(-1) == 3 and pos.size(1) == 4:
                # (total_residues, 4, 3) -> 取 CA（索引0）
                ca = pos[:, 0, :]          # (total_residues, 3)
            elif pos.size(-1) == 3:
                # (batch, max_len, 3) -> 展平
                ca = pos.reshape(-1, 3)
            else:
                raise ValueError(f"Unsupported 3D pos shape: {pos.shape}")
        elif pos.dim() == 4:
            # 形状可能是 (batch, max_len, 4, 3)
            if pos.size(-2) == 4 and pos.size(-1) == 3:
                ca = pos[..., 0, :]        # (batch, max_len, 3)
                ca = ca.reshape(-1, 3)
            else:
                raise ValueError(f"Unsupported 4D pos shape: {pos.shape}")
        else:
            raise ValueError(f"Unsupported pos dimension: {pos.dim()}")

        # ---------- 2. 分序列计算距离矩阵 ----------
        if batch_index is None:
            dist = torch.cdist(ca, ca, p=2)          # (L, L)
            if self.use_mlp:
                bias = self.distance_mlp(dist.unsqueeze(-1)).squeeze(-1)
            else:
                bias = -dist * self.scale
            return [bias]

        unique_batches = batch_index.unique()
        pair_biases = []
        for b in unique_batches:
            mask = (batch_index == b)
            ca_b = ca[mask]                         # (L_i, 3)
            dist = torch.cdist(ca_b, ca_b, p=2)     # (L_i, L_i)
            if self.use_mlp:
                bias = self.distance_mlp(dist.unsqueeze(-1)).squeeze(-1)
            else:
                bias = -dist * self.scale
            # 确保 bias 是 2D
            if bias.dim() != 2:
                raise RuntimeError(f"Expected 2D bias, got shape {bias.shape} for batch {b}")
            pair_biases.append(bias)
        return pair_biases