# src/losses/cswcl_loss.py
"""
论文 Sec. III-C 的代价敏感加权对比损失（CSWCL），对应：
    Eq.(10)  L^FC_cswcl
    Eq.(11)  L^HOFC_cswcl
    Eq.(12)  L_cswcl = L^FC_cswcl + L^HOFC_cswcl
    Eq.(19)  phi_j^FC / phi_j^HOFC（负样本权重）
"""

import torch
import torch.nn.functional as F


def _compute_phi_weights(
    z: torch.Tensor,        # [N, D]
    p: torch.Tensor,        # [K, D]
    y: torch.Tensor,        # [N]
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Eq.(19)：phi_j = 1 / ||z_j - p_{y_j}||_2

    ★ 注意：调用时 z 和 p 都应该是 detach 过的。
    """
    p_assigned = p[y]                                  # [N, D]
    dist = torch.norm(z - p_assigned, p=2, dim=1)     # [N]
    phi = 1.0 / (dist + eps)                           # [N]
    return phi


def compute_cswcl_loss(
    z_fc: torch.Tensor,        # [N_l, D]
    z_hofc: torch.Tensor,      # [N_l, D]
    y: torch.Tensor,           # [N_l]
    p_fc: torch.Tensor,        # [K, D]
    p_hofc: torch.Tensor,      # [K, D]
    tau: float   = 0.2,
    eps: float   = 1e-8,
) -> torch.Tensor:
    """
    实现论文 Eq.(10)-(12) 的代价敏感加权对比损失。

    ★ 关键修复：
    1. phi 权重完全 detach（z 和 p 都 detach），只作为常数权重
       - 如果不 detach z，梯 度会驱使 z 远离原型来减小 phi，
         与 L_proto 的梯度方向完全矛盾，两个 loss 互相抵消
    # 2. w_i 做归一化，防止数值不稳定(？？？不用
    3. phi 做 clamp，防止极端值
    """
    N      = z_fc.size(0)
    device = z_fc.device

    # ── 1. 余弦相似度归一化 ──
    z_fc_n   = F.normalize(z_fc,   dim=1, eps=eps)
    z_hofc_n = F.normalize(z_hofc, dim=1, eps=eps)

    # ── 2. 正样本相似度 s+_i ──
    s_pos = (z_fc_n * z_hofc_n).sum(dim=1)          # [N]

    # ── 3. 全对相似度矩阵 ──
    sim_fc_hofc = z_fc_n   @ z_hofc_n.T             # [N, N]
    sim_hofc_fc = z_hofc_n @ z_fc_n.T               # [N, N]

    # ── 4. 代价敏感权重 w_i = N^l / N^l_{y_i} ──
    num_classes  = p_fc.size(0)
    class_counts = torch.zeros(num_classes, dtype=z_fc.dtype, device=device)
    for k in range(num_classes):
        class_counts[k] = (y == k).sum().float()
    class_counts = class_counts.clamp(min=1.0)

    # base_w = N / class_counts[y]  # [N]
    # w = torch.pow(base_w, 1.1)  # ★ 核心修改
    w = N / class_counts[y]                          # [N]

    # ── 5. phi 权重 (完全 detach！不参与反向传播) ──
    # ★★★ 核心修复：z 和 p 都 detach ★★★
    # 论文中 phi 只是一个权重系数，不应该通过它来更新编码器参数
    phi_fc   = _compute_phi_weights(z_fc.detach(),   p_fc.detach(),   y, eps=eps)
    phi_hofc = _compute_phi_weights(z_hofc.detach(), p_hofc.detach(), y, eps=eps)

    # clamp 防止极端值导致数值爆炸
    # phi_fc   = phi_fc.clamp(max=50.0)
    # phi_hofc = phi_hofc.clamp(max=50.0)

    # ── 6. 归一化因子 ──
    E_fc   = 1.0 / (phi_fc.sum()   + eps)
    E_hofc = 1.0 / (phi_hofc.sum() + eps)

    # ── 7. L^FC_cswcl (Eq.10) ──
    exp_pos = torch.exp(s_pos / tau)                  # [N]

    mask_off_diag = ~torch.eye(N, dtype=torch.bool, device=device)

    weighted_neg_fc = (
        torch.exp(sim_fc_hofc / tau)
        * phi_hofc.unsqueeze(0)
        * mask_off_diag.float()
    ).sum(dim=1)                                       # [N]

    denom_fc = exp_pos + E_hofc * weighted_neg_fc
    denom_fc = denom_fc.clamp(min=eps)

    L_fc = -(w * (s_pos / tau - torch.log(denom_fc))).mean()

    # ── 8. L^HOFC_cswcl (Eq.11) ──
    weighted_neg_hofc = (
        torch.exp(sim_hofc_fc / tau)
        * phi_fc.unsqueeze(0)
        * mask_off_diag.float()
    ).sum(dim=1)

    denom_hofc = exp_pos + E_fc * weighted_neg_hofc
    denom_hofc = denom_hofc.clamp(min=eps)

    L_hofc = -(w * (s_pos / tau - torch.log(denom_hofc))).mean()

    # ── 9. L_cswcl ──
    L_cswcl = L_fc + L_hofc

    return L_cswcl