# src/losses/prototype_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def split_support_query_per_class(
    y: torch.Tensor,
    num_classes: int,
    seed: int = 42,
    n_query: int = None,       # ← 新增：每类 query 数量，None 表示自适应
    query_ratio: float = 0.15, # ← 每类取 15% 作为 query（自适应模式）
):
    """
    更贴论文的划分方式：
    对每一类 k：
      - Q_k 只包含 1 个 query 样本
      - S_k 包含其余所有训练样本

    返回：
      support_idx: list[Tensor]
      query_idx:   list[Tensor]
    """
    device = y.device
    generator = torch.Generator()
    generator.manual_seed(seed)

    support_idx, query_idx = [], []

    for k in range(num_classes):
        idx = torch.where(y == k)[0].cpu()  # 先放到 cpu 方便 randperm

        if len(idx) == 0:
            support_idx.append(torch.empty(0, dtype=torch.long, device=device))
            query_idx.append(torch.empty(0, dtype=torch.long, device=device))
            continue

        # 打乱该类样本顺序
        perm = idx[torch.randperm(len(idx), generator=generator)]

        # 若该类只有 1 个样本：无法同时分 support/query
        # 这里退化处理：该样本放 support，query 为空
        if len(perm) == 1:
            support_idx.append(perm.to(device))
            query_idx.append(torch.empty(0, dtype=torch.long, device=device))
            continue

        # 论文设定：每类 1 个 query，其余全部 support
        q = perm[:1].to(device)     # [1]
        s = perm[1:].to(device)     # [Nk-1]

        support_idx.append(s)
        query_idx.append(q)

    return support_idx, query_idx


def compute_prototypes(
    z_fc: torch.Tensor,
    z_hofc: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    seed: int = 42,
):
    """
    对应论文 Eq.(13a)(13b)：
        p_k^FC   = mean_{z_i in S_k} z_i^FC
        p_k^HOFC = mean_{z_i in S_k} z_i^HOFC
    """
    support_idx, query_idx = split_support_query_per_class(
        y=y,
        num_classes=num_classes,
        seed=seed
    )

    emb_dim = z_fc.size(1)

    p_fc = torch.zeros(num_classes, emb_dim, device=z_fc.device)
    p_hofc = torch.zeros(num_classes, emb_dim, device=z_fc.device)
    valid_class_mask = torch.zeros(num_classes, dtype=torch.bool, device=z_fc.device)

    for k in range(num_classes):
        idx = support_idx[k]
        if len(idx) == 0:
            continue

        p_fc[k] = z_fc[idx].mean(dim=0)
        p_hofc[k] = z_hofc[idx].mean(dim=0)
        valid_class_mask[k] = True

    return p_fc, p_hofc, support_idx, query_idx, valid_class_mask


class SharedDistanceLayer(nn.Module):
    """
    更贴论文 Eq.(14)(15) 的 shared distance layer h(·)：

        d_i,k^FC   = diff(z_i^FC,   p_k^FC)   = z_i^FC   - p_k^FC
        d_i,k^HOFC = diff(z_i^HOFC, p_k^HOFC) = z_i^HOFC - p_k^HOFC

        d_i^FC     = concat_k d_i,k^FC
        d_i^HOFC   = concat_k d_i,k^HOFC
        d_i        = [d_i^FC, d_i^HOFC]

    注意：
    这里不做 L2 normalize，保持更贴近论文原始定义。
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        z_fc: torch.Tensor,    # [N, D]
        z_hofc: torch.Tensor,  # [N, D]
        p_fc: torch.Tensor,    # [K, D]
        p_hofc: torch.Tensor,  # [K, D]
    ) -> torch.Tensor:
        # [N, K, D]
        diff_fc = z_fc.unsqueeze(1) - p_fc.unsqueeze(0)
        diff_hofc = z_hofc.unsqueeze(1) - p_hofc.unsqueeze(0)

        # [N, K*D]
        d_fc = diff_fc.reshape(z_fc.size(0), -1)
        d_hofc = diff_hofc.reshape(z_hofc.size(0), -1)

        # [N, 2*K*D]
        d = torch.cat([d_fc, d_hofc], dim=1)
        return d


class SharedDistanceClassifier(nn.Module):
    """
    prototype matching 分类器
    在 logits 上加入 temperature 缩放
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.h = SharedDistanceLayer()
        self.temperature = temperature

    def forward(
        self,
        q_fc: torch.Tensor,    # [K_valid, D]
        q_hofc: torch.Tensor,  # [K_valid, D]
        p_fc: torch.Tensor,    # [K_valid, D]
        p_hofc: torch.Tensor,  # [K_valid, D]
    ):
        # prototype 的距离表示
        Hs = self.h(p_fc, p_hofc, p_fc, p_hofc)   # [K, 2*K*D]

        # query 的距离表示
        Hq = self.h(q_fc, q_hofc, p_fc, p_hofc)   # [K, 2*K*D]

        # logits 加 temperature
        logits = (Hq @ Hs.T) / self.temperature   # [K, K]
        prob = F.softmax(logits, dim=1)

        return logits, prob, Hs, Hq


def prototype_loss(
    z_fc: torch.Tensor,
    z_hofc: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    proto_classifier: nn.Module,
    seed: int = 42,
):
    """
    更贴论文 Eq.(13)-(18) 的原型损失：

      1) 每类划分 support / query（每类 1 个 query）
      2) support 均值 -> prototype
      3) h_k = h(p_k), h_k^Q = h(q_k)
      4) e_hat_k = softmax((H^S)^T · h_k^Q)
      5) L_proto = CrossEntropy(one-hot(k), e_hat_k)

    返回：
      loss, p_fc, p_hofc
    """
    p_fc, p_hofc, support_idx, query_idx, valid_class_mask = compute_prototypes(
        z_fc=z_fc,
        z_hofc=z_hofc,
        y=y,
        num_classes=num_classes,
        seed=seed
    )

    # 收集每个有效类别的唯一 query
    q_fc_list, q_hofc_list, q_label_list = [], [], []

    for k in range(num_classes):
        idx = query_idx[k]
        if len(idx) == 0:
            continue

        # 按论文，每类 query 只有 1 个
        q_idx = idx[0]
        q_fc_list.append(z_fc[q_idx])
        q_hofc_list.append(z_hofc[q_idx])
        q_label_list.append(k)

    # 如果没有任何 query，可返回 0 loss
    if len(q_label_list) == 0:
        loss = z_fc.sum() * 0.0
        return loss, p_fc, p_hofc

    q_fc_t = torch.stack(q_fc_list, dim=0)         # [K_valid, D]
    q_hofc_t = torch.stack(q_hofc_list, dim=0)     # [K_valid, D]
    q_labels = torch.tensor(q_label_list, dtype=torch.long, device=z_fc.device)

    # 只保留有效类别（防止某类没有 support）
    valid_idx = torch.where(valid_class_mask)[0]
    p_fc_valid = p_fc[valid_idx]
    p_hofc_valid = p_hofc[valid_idx]

    # 若并非所有类别都有效，需要把标签重新映射到 [0, K_valid-1]
    if len(valid_idx) != num_classes:
        mapper = {int(c): i for i, c in enumerate(valid_idx.tolist())}

        keep = torch.tensor(
            [int(lbl.item()) in mapper for lbl in q_labels],
            dtype=torch.bool,
            device=z_fc.device
        )

        q_fc_t = q_fc_t[keep]
        q_hofc_t = q_hofc_t[keep]
        q_labels_raw = q_labels[keep]

        if len(q_labels_raw) == 0:
            loss = z_fc.sum() * 0.0
            return loss, p_fc, p_hofc

        q_labels_mapped = torch.tensor(
            [mapper[int(lbl.item())] for lbl in q_labels_raw],
            dtype=torch.long,
            device=z_fc.device
        )

        logits, _, _, _ = proto_classifier(
            q_fc=q_fc_t,
            q_hofc=q_hofc_t,
            p_fc=p_fc_valid,
            p_hofc=p_hofc_valid
        )
        loss = F.cross_entropy(logits, q_labels_mapped)
        return loss, p_fc, p_hofc

    # 正常路径：所有类别都有效
    logits, _, _, _ = proto_classifier(
        q_fc=q_fc_t,
        q_hofc=q_hofc_t,
        p_fc=p_fc,
        p_hofc=p_hofc
    )
    loss = F.cross_entropy(logits, q_labels)

    return loss, p_fc, p_hofc