# src/losses/contrastive_loss.py
import torch
import torch.nn.functional as F


def compute_contrastive_loss(
    z_fc: torch.Tensor,
    z_hofc: torch.Tensor,
    tau: float = 0.2,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    普通跨视图对比学习损失（对应 CP-GCN）。

    正样本对：
        (z_fc_i, z_hofc_i)

    负样本对：
        (z_fc_i, z_hofc_j), j != i
        (z_hofc_i, z_fc_j), j != i

    不使用：
        - 类别代价敏感权重 w_i
        - 负样本原型权重 phi_j
    """
    N = z_fc.size(0)
    device = z_fc.device

    z_fc_n   = F.normalize(z_fc, dim=1, eps=eps)
    z_hofc_n = F.normalize(z_hofc, dim=1, eps=eps)

    s_pos = (z_fc_n * z_hofc_n).sum(dim=1)   # [N]

    sim_fc_hofc = z_fc_n @ z_hofc_n.T        # [N, N]
    sim_hofc_fc = z_hofc_n @ z_fc_n.T        # [N, N]

    exp_pos = torch.exp(s_pos / tau)         # [N]

    mask_off_diag = ~torch.eye(N, dtype=torch.bool, device=device)

    neg_fc = (
        torch.exp(sim_fc_hofc / tau) * mask_off_diag.float()
    ).sum(dim=1)  # [N]

    neg_hofc = (
        torch.exp(sim_hofc_fc / tau) * mask_off_diag.float()
    ).sum(dim=1)  # [N]

    denom_fc   = (exp_pos + neg_fc).clamp(min=eps)
    denom_hofc = (exp_pos + neg_hofc).clamp(min=eps)

    l_fc   = -(s_pos / tau - torch.log(denom_fc)).mean()
    l_hofc = -(s_pos / tau - torch.log(denom_hofc)).mean()

    loss = l_fc + l_hofc
    return loss