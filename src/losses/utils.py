# src/losses/utils.py
import torch
import torch.nn.functional as F


def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    计算两批向量的余弦相似度
    a, b: [..., D]
    return: [...]
    """
    a = F.normalize(a, dim=-1, eps=eps)
    b = F.normalize(b, dim=-1, eps=eps)
    return torch.sum(a * b, dim=-1)


def pairwise_cosine_sim(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    A: [N, D], B: [M, D]
    return: [N, M]
    """
    A = F.normalize(A, dim=-1, eps=eps)
    B = F.normalize(B, dim=-1, eps=eps)
    return A @ B.T