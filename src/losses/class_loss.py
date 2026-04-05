# src/losses/class_loss.py
import torch
import torch.nn.functional as F


def class_balanced_cross_entropy_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    论文 Eq.(7): 按类别加权的交叉熵

    L_class = - sum_k (1/N_k^l) * sum_{i in I_k} y_{i,k} * ln(y_hat_{i,k})

    ★ 改回论文原本的加权 CE，不用 Focal Loss。
    理由：
    1. Focal Loss 自带难例挖掘能力，会削弱 CSWCL 的边际效果
    2. 论文的消融实验是在加权 CE 基础上做的，换了基线损失后消融不公平
    3. gamma 参数保留但不使用，保持接口兼容
    """
    counts = torch.bincount(y, minlength=num_classes).float().to(logits.device)
    counts = counts.clamp(min=1.0)

    # 论文权重: 1/N_k，归一化保持量级
    alpha = 1.0 / counts               # [K]
    alpha = alpha / alpha.mean()        # 归一化

    log_probs = F.log_softmax(logits, dim=1)            # [N, K]
    log_p_t   = log_probs.gather(1, y.unsqueeze(1)).squeeze(1)  # [N]
    alpha_t   = alpha[y]                                 # [N]

    loss = -(alpha_t * log_p_t).mean()
    return loss