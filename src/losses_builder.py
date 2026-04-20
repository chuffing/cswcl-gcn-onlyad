# src/losses_builder.py
import torch
import torch.nn as nn

from src.losses.class_loss import class_balanced_cross_entropy_loss
from src.losses.prototype_loss import prototype_loss
from src.losses.contrastive_loss import compute_contrastive_loss
from src.losses.weighted_contrastive_loss import compute_weighted_contrastive_loss
from src.losses.cswcl_loss import compute_cswcl_loss


def total_loss(
    outputs: dict,
    y: torch.Tensor,
    num_classes: int,
    proto_classifier: nn.Module,
    n_query: int = None,
    query_ratio: float = 0.15,
    mode: str = "cswcl",
    p_lambda_proto=0.03,
    lambda_cl=0.01,
    cp_lambda_proto=0.03,
    lambda_wcl=0.01,
    wcl_lambda_proto=0.03,
    lambda_cswcl=0.03,
    cswcl_lambda_proto=0.25,
    temperature: float = 0.2,
    focal_gamma: float = 2.0,
    seed: int = 42,
):
    """
    根据 ablation mode 组装总损失。

    mode:
        mv    -> L_class
        p     -> L_class + lambda_proto * L_proto
        cp    -> L_class + lambda_cswcl * L_cl   + lambda_proto * L_proto
        wcl   -> L_class + lambda_cswcl * L_wcl  + lambda_proto * L_proto
        cswcl -> L_class + lambda_cswcl * L_cswcl + lambda_proto * L_proto
    """
    mode = mode.lower().strip()

    logits = outputs["logits"]
    z_fc   = outputs["z_fc"]
    z_hofc = outputs["z_hofc"]

    # 1) 分类损失（所有模式都要）
    l_class = class_balanced_cross_entropy_loss(
        logits, y, num_classes, gamma=focal_gamma
    )

    # 2) 原型损失 + 原型（除了 mv 外，其余模式都需要）
    if mode == "mv":
        l_proto = torch.zeros((), device=logits.device)
        p_fc    = None
        p_hofc  = None
    else:
        l_proto, p_fc, p_hofc = prototype_loss(
            z_fc=z_fc,
            z_hofc=z_hofc,
            y=y,
            num_classes=num_classes,
            proto_classifier=proto_classifier,
            seed=seed,
            n_query=n_query,
            query_ratio=query_ratio,
        )
        l_proto = torch.clamp(l_proto, max=2.0)

    # 3) 各种 contrastive 组件
    l_cl    = torch.zeros((), device=logits.device)
    l_wcl   = torch.zeros((), device=logits.device)
    l_cswcl = torch.zeros((), device=logits.device)

    if mode == "cp":
        l_cl = compute_contrastive_loss(
            z_fc=z_fc,
            z_hofc=z_hofc,
            tau=temperature,
        )
    elif mode == "wcl":
        l_wcl = compute_weighted_contrastive_loss(
            z_fc=z_fc,
            z_hofc=z_hofc,
            y=y,
            p_fc=p_fc,
            p_hofc=p_hofc,
            tau=temperature,
        )
    elif mode == "cswcl":
        l_cswcl = compute_cswcl_loss(
            z_fc=z_fc,
            z_hofc=z_hofc,
            y=y,
            p_fc=p_fc,
            p_hofc=p_hofc,
            tau=temperature,
        )

    if mode == "mv":
        loss = l_class
    elif mode == "p":
        loss = l_class + p_lambda_proto * l_proto
    elif mode == "cp":
        loss = l_class + lambda_cl * l_cl + cp_lambda_proto * l_proto
    elif mode == "wcl":
        loss = l_class + lambda_wcl * l_wcl + wcl_lambda_proto * l_proto
    elif mode == "cswcl":
        loss = l_class + lambda_cswcl * l_cswcl + cswcl_lambda_proto * l_proto
    else:
        raise ValueError(f"不支持的 mode: {mode}")

    loss_dict = {
        "loss":    loss.detach(),
        "l_class": l_class.detach(),
        "l_proto": l_proto.detach(),
        "l_cl":    l_cl.detach(),
        "l_wcl":   l_wcl.detach(),
        "l_cswcl": l_cswcl.detach(),
    }
    return loss, loss_dict