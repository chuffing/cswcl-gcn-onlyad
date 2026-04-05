# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """
    单层 GCN：H^(l) = act( A_norm @ H^(l-1) @ W )
    """
    def __init__(self, in_dim, out_dim, dropout=0.0, activate=True):
        super().__init__()
        self.linear   = nn.Linear(in_dim, out_dim, bias=True)
        self.dropout  = nn.Dropout(dropout)
        self.activate = activate

    def forward(self, x, a_norm):
        x = self.dropout(x)
        x = torch.matmul(a_norm, x)
        x = self.linear(x)
        if self.activate:
            x = F.relu(x)
        return x


class ThreeLayerGCN(nn.Module):
    """
    3 层 GCN 编码器。
    """
    def __init__(self, in_dim, hidden_dim, emb_dim, dropout=0.3):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim,       hidden_dim, dropout=dropout, activate=True)
        self.gcn2 = GCNLayer(hidden_dim,   hidden_dim, dropout=dropout, activate=True)
        self.gcn3 = GCNLayer(hidden_dim,   emb_dim,    dropout=dropout, activate=True)

        # BatchNorm 稳定训练
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, a_norm):
        h1 = self.bn1(self.gcn1(x, a_norm))
        h2 = self.bn2(self.gcn2(h1, a_norm))
        z  = self.gcn3(h2, a_norm)
        return z


class TwoViewAMAttention(nn.Module):
    """
    AM-GCN 风格的两视图 attention：
        omega_i^FC   = q^T tanh(W z_i^FC + b)
        omega_i^HOFC = q^T tanh(W z_i^HOFC + b)
        alpha = softmax([omega^FC, omega^HOFC])
        z_global = alpha_fc * z_fc + alpha_hofc * z_hofc
    """
    def __init__(self, emb_dim, att_dim=None):
        super().__init__()
        if att_dim is None:
            att_dim = emb_dim

        self.W = nn.Linear(emb_dim, att_dim, bias=True)
        self.q = nn.Parameter(torch.empty(att_dim, 1))
        nn.init.xavier_uniform_(self.q)

    def _score(self, z):
        h     = torch.tanh(self.W(z))       # [N, att_dim]
        omega = torch.matmul(h, self.q)     # [N, 1]，表示这个视图对当前样本的重要程度打分
        return omega

    def forward(self, z_fc, z_hofc):
        omega_fc   = self._score(z_fc)      # [N, 1]
        omega_hofc = self._score(z_hofc)    # [N, 1]

        omega = torch.cat([omega_fc, omega_hofc], dim=1)  # [N, 2]
        alpha = F.softmax(omega, dim=1)                   # [N, 2]

        z_global = alpha[:, 0:1] * z_fc + alpha[:, 1:2] * z_hofc
        return z_global, alpha


class ClassifierHead(nn.Module):
    def __init__(self, emb_dim, num_classes, dropout=0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(emb_dim, num_classes)

    def forward(self, z):
        return self.fc(self.dropout(z))  # 返回原始分数


class CSWCLGCN(nn.Module):
    """
    完整模型：FC-GCN + HOFC-GCN + 两视图 attention + classifier
    """
    def __init__(self, in_dim, hidden_dim, emb_dim, num_classes,
                 dropout=0.3, att_dim=None):
        super().__init__()

        self.encoder_fc   = ThreeLayerGCN(in_dim, hidden_dim, emb_dim, dropout)
        self.encoder_hofc = ThreeLayerGCN(in_dim, hidden_dim, emb_dim, dropout)
        self.fusion       = TwoViewAMAttention(emb_dim, att_dim)
        self.classifier   = ClassifierHead(emb_dim, num_classes, dropout)

    def forward(self, x_fc, x_hofc, a_fc_norm, a_hofc_norm):
        z_fc   = self.encoder_fc(x_fc,   a_fc_norm)
        z_hofc = self.encoder_hofc(x_hofc, a_hofc_norm)

        z_global, alpha = self.fusion(z_fc, z_hofc)
        logits = self.classifier(z_global)

        return {
            "z_fc":    z_fc,
            "z_hofc":  z_hofc,
            "z_global": z_global,
            "alpha":   alpha,
            "logits":  logits
        }