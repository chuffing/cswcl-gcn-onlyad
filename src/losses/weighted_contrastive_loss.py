# src/losses/weighted_contrastive_loss.py
import torch
import torch.nn.functional as F

def print_phi_stats(phi: torch.Tensor, name: str = "phi"):
    """
    打印 phi 的分布统计信息
    """
    phi_det = phi.detach().float().view(-1)

    q = torch.quantile(
        phi_det,
        torch.tensor([0.5, 0.9, 0.95, 0.99], device=phi_det.device)
    )

    print(f"[{name}] "
          f"min={phi_det.min().item():.4f}, "
          f"max={phi_det.max().item():.4f}, "
          f"mean={phi_det.mean().item():.4f}, "
          f"std={phi_det.std().item():.4f}, "
          f"median={q[0].item():.4f}, "
          f"p90={q[1].item():.4f}, "
          f"p95={q[2].item():.4f}, "
          f"p99={q[3].item():.4f}")


def _compute_phi_weights(
    z: torch.Tensor,
    p: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    phi_j = 1 / || z_j - p_{y_j} ||_2
    """
    p_assigned = p[y]                              # [N, D]
    dist = torch.norm(z - p_assigned, p=2, dim=1) # [N]
    phi = 1.0 / (dist + eps)
    return phi


def compute_weighted_contrastive_loss(
    z_fc: torch.Tensor,
    z_hofc: torch.Tensor,
    y: torch.Tensor,
    p_fc: torch.Tensor,
    p_hofc: torch.Tensor,
    tau: float = 0.2,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    加权对比学习损失（对应 WCL-GCN）。

    使用：
        - 负样本权重 phi_j

    不使用：
        - 类别代价敏感权重 w_i

    ★ 修复：phi 完全 detach，不参与反向传播
    """
    N = z_fc.size(0)
    device = z_fc.device

    z_fc_n   = F.normalize(z_fc, dim=1, eps=eps)
    z_hofc_n = F.normalize(z_hofc, dim=1, eps=eps)

    s_pos = (z_fc_n * z_hofc_n).sum(dim=1)    # [N]

    sim_fc_hofc = z_fc_n @ z_hofc_n.T         # [N, N]
    sim_hofc_fc = z_hofc_n @ z_fc_n.T         # [N, N]

    # ★★★ 核心修复：z 和 p 都 detach ★★★
    phi_fc   = _compute_phi_weights(z_fc.detach(),   p_fc.detach(),   y, eps=eps)
    phi_hofc = _compute_phi_weights(z_hofc.detach(), p_hofc.detach(), y, eps=eps)

    # 先看 clamp 前的原始分布
    # print_phi_stats(phi_fc, "phi_fc_before_clamp")
    # print_phi_stats(phi_hofc, "phi_hofc_before_clamp")

    # clamp 防止极端值，
    # phi_fc = phi_fc.clamp(max=50.0)
    # phi_hofc = phi_hofc.clamp(max=50.0)
    # 不过这个50完全没必要啊！phi比较小

    # 再看 clamp 后的分布
    # print_phi_stats(phi_fc, "phi_fc_after_clamp")
    # print_phi_stats(phi_hofc, "phi_hofc_after_clamp")

    E_fc   = 1.0 / (phi_fc.sum() + eps)
    E_hofc = 1.0 / (phi_hofc.sum() + eps)

    exp_pos = torch.exp(s_pos / tau)

    mask_off_diag = ~torch.eye(N, dtype=torch.bool, device=device)

    weighted_neg_fc = (
        torch.exp(sim_fc_hofc / tau)
        * phi_hofc.unsqueeze(0)
        * mask_off_diag.float()
    ).sum(dim=1)

    weighted_neg_hofc = (
        torch.exp(sim_hofc_fc / tau)
        * phi_fc.unsqueeze(0)
        * mask_off_diag.float()
    ).sum(dim=1)

    denom_fc   = (exp_pos + E_hofc * weighted_neg_fc).clamp(min=eps)
    denom_hofc = (exp_pos + E_fc   * weighted_neg_hofc).clamp(min=eps)

    l_fc   = -(s_pos / tau - torch.log(denom_fc)).mean()
    l_hofc = -(s_pos / tau - torch.log(denom_hofc)).mean()

    loss = l_fc + l_hofc
    return loss