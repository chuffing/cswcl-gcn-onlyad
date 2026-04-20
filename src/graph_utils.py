# src/graph_utils.py
import numpy as np
import torch
from sklearn.metrics import pairwise_distances


def estimate_sigma(X: np.ndarray, method: str = "median") -> float:
    """
    估计高斯核宽度 sigma。
    ★ X 只应传训练集，防止 test 数据影响 sigma 估计。
    """
    D = pairwise_distances(X, metric="euclidean")  # D[i, j]表示第i个样本和第j个样本的欧氏距离
    triu = D[np.triu_indices_from(D, k=1)]

    if method == "median":
        sigma = float(np.median(triu))
    elif method == "mean":
        sigma = float(np.mean(triu))
    elif method == "percentile25":
        sigma = float(np.percentile(triu, 25))
    else:
        raise ValueError(f"Unsupported sigma method: {method}")

    if sigma < 1e-8:
        sigma = 1.0
    return sigma


def compute_similarity_matrix(X: np.ndarray, sigma: float) -> np.ndarray:
    """
    Sim(x_i, x_j) = exp( -||x_i - x_j||^2 / (2*sigma^2) )
    """
    D = pairwise_distances(X, metric="euclidean")
    S = np.exp(-(D ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
    return S


def compute_sex_gate(sex: np.ndarray) -> np.ndarray:
    """
    categorical phenotype:
        gamma(sex_i, sex_j) = 1  if sex_i == sex_j 且均非 -1
                            = 0  otherwise
    """
    sex = np.asarray(sex)
    same = (sex[:, None] == sex[None, :]).astype(np.float32)
    unknown = (sex == -1).astype(np.float32)
    mask_bad = np.outer(unknown, np.ones(len(sex))) + np.outer(np.ones(len(sex)), unknown)
    same[mask_bad > 0] = 0.0
    return same


def compute_age_gate(age: np.ndarray, threshold: float = 5.0) -> np.ndarray:
    """
    gamma(age_i, age_j) = 1 if |age_i - age_j| < threshold 且二者都非负
                        = 0 otherwise
    """
    age = np.asarray(age, dtype=np.float32)
    diff = np.abs(age[:, None] - age[None, :])
    gate = (diff < threshold).astype(np.float32)

    unknown = (age < 0).astype(np.float32)
    mask_bad = np.outer(unknown, np.ones(len(age))) + np.outer(np.ones(len(age)), unknown)
    gate[mask_bad > 0] = 0.0
    return gate


def compute_edu_gate(edu: np.ndarray, threshold: float = 2.0) -> np.ndarray:
    """
    教育程度相似门控：
        gamma(edu_i, edu_j) = 1 if |edu_i - edu_j| < threshold 且二者都非负
                            = 0 otherwise
    """
    edu = np.asarray(edu, dtype=np.float32)
    diff = np.abs(edu[:, None] - edu[None, :])
    gate = (diff < threshold).astype(np.float32)

    unknown = (edu < 0).astype(np.float32)
    mask_bad = np.outer(unknown, np.ones(len(edu))) + np.outer(np.ones(len(edu)), unknown)
    gate[mask_bad > 0] = 0.0
    return gate


def compute_site_gate(site: np.ndarray, mode: str = "cross") -> np.ndarray:
    """
    站点门控：
    - cross: 只保留不同站点之间的边
    - same:  只保留相同站点之间的边
    """
    site = np.asarray(site, dtype=object)
    site_norm = np.array([str(x).strip() for x in site], dtype=object)
    unknown = np.isin(site_norm, ["", "-1", "nan", "none", "None"])

    same = (site_norm[:, None] == site_norm[None, :]).astype(np.float32)
    if mode == "cross":
        gate = 1.0 - same
    elif mode == "same":
        gate = same
    else:
        raise ValueError(f"Unsupported site gate mode: {mode}")

    mask_bad = np.outer(unknown, np.ones(len(site_norm), dtype=bool)) | np.outer(
        np.ones(len(site_norm), dtype=bool), unknown
    )
    gate[mask_bad] = 0.0
    return gate.astype(np.float32)


def knn_sparsify(A: np.ndarray, k: int) -> np.ndarray:
    """
    k-NN 稀疏化：每个节点只保留相似度最高的 k 个非自身邻居。
    """
    N = A.shape[0]
    diag = np.diag(A).copy()

    A_off = A.copy()
    np.fill_diagonal(A_off, 0.0)

    A_knn = np.zeros_like(A_off)
    for i in range(N):
        row = A_off[i]
        nonzero = np.where(row > 1e-10)[0]
        if len(nonzero) == 0:
            continue
        actual_k = min(k, len(nonzero))
        top_k = nonzero[np.argsort(row[nonzero])[-actual_k:]]
        A_knn[i, top_k] = row[top_k]

    A_sym = np.maximum(A_knn, A_knn.T)
    np.fill_diagonal(A_sym, diag)
    return A_sym.astype(np.float32)


def build_adjacency(
    X: np.ndarray,
    sex: np.ndarray = None,
    age: np.ndarray = None,
    edu: np.ndarray = None,
    site: np.ndarray = None,
    sigma: float = None,
    sigma_method: str = "median",
    add_self_loop: bool = True,
    knn: int = None,
    X_for_sigma: np.ndarray = None,
    use_sex_gate: bool = True,
    use_age_gate: bool = True,
    use_edu_gate: bool = True,
    use_site_gate: bool = False,
    site_gate_mode: str = "cross",
    age_threshold: float = 5.0,
    edu_threshold: float = 2.0,
    edu_weight: float = 0.10
):
    """
    构图：
        A = Sim(X) * gamma(sex) * gamma(age) + edu_weight * gamma(edu)

    其中 EDU 作为软加成，而不是继续乘法硬筛，
    防止图过稀。
    """
    if sigma is None:
        ref = X_for_sigma if X_for_sigma is not None else X
        sigma = estimate_sigma(ref, method=sigma_method)

    S = compute_similarity_matrix(X, sigma=sigma)

    G = np.ones_like(S, dtype=np.float32)

    if use_sex_gate and sex is not None:
        G = G * compute_sex_gate(sex)

    if use_age_gate and age is not None:
        G = G * compute_age_gate(age, threshold=age_threshold)

    if use_site_gate and site is not None:
        G = G * compute_site_gate(site, mode=site_gate_mode)

    A = (S * G).astype(np.float32)

    # ★ EDU 软加成
    if use_edu_gate and edu is not None:
        edu_gate = compute_edu_gate(edu, threshold=edu_threshold)
        A = A + edu_weight * edu_gate.astype(np.float32)

    # k-NN 稀疏化
    if knn is not None and knn > 0:
        A = knn_sparsify(A, k=knn)

    if add_self_loop:
        np.fill_diagonal(A, 1.0)
    else:
        np.fill_diagonal(A, 0.0)

    return A, sigma


def normalize_adj_numpy(A: np.ndarray) -> np.ndarray:
    """GCN 标准对称归一化：A_norm = D^{-1/2} A D^{-1/2}"""
    A = A.astype(np.float32)
    deg = A.sum(axis=1)
    deg[deg < 1e-8] = 1.0
    d_inv_sqrt = np.power(deg, -0.5)
    return (A * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]).astype(np.float32)


def numpy_to_torch_tensor(X: np.ndarray, device="cpu", dtype=torch.float32):
    return torch.tensor(X, dtype=dtype, device=device)


def build_two_view_graphs(
    X_fc: np.ndarray,
    X_hofc: np.ndarray,
    sex: np.ndarray = None,
    age: np.ndarray = None,
    edu: np.ndarray = None,
    site: np.ndarray = None,
    sigma_fc: float = None,
    sigma_hofc: float = None,
    sigma_method: str = "median",
    knn: int = 10,
    X_fc_train: np.ndarray = None,
    X_hofc_train: np.ndarray = None,
    use_sex_gate: bool = True,
    use_age_gate: bool = True,
    use_edu_gate: bool = True,
    use_site_gate: bool = False,
    site_gate_mode: str = "cross",
    age_threshold: float = 5.0,
    edu_threshold: float = 2.0,
    edu_weight: float = 0.10,
    device: str = "cpu"
):
    """
    为 FC / HOFC 两个视图分别构图并做 GCN 归一化。
    """
    A_fc, sigma_fc_used = build_adjacency(
        X=X_fc,
        sex=sex,
        age=age,
        edu=edu,
        site=site,
        sigma=sigma_fc,
        sigma_method=sigma_method,
        knn=knn,
        X_for_sigma=X_fc_train,
        add_self_loop=True,
        use_sex_gate=use_sex_gate,
        use_age_gate=use_age_gate,
        use_edu_gate=use_edu_gate,
        use_site_gate=use_site_gate,
        site_gate_mode=site_gate_mode,
        age_threshold=age_threshold,
        edu_threshold=edu_threshold,
        edu_weight=edu_weight
    )

    A_hofc, sigma_hofc_used = build_adjacency(
        X=X_hofc,
        sex=sex,
        age=age,
        edu=edu,
        site=site,
        sigma=sigma_hofc,
        sigma_method=sigma_method,
        knn=knn,
        X_for_sigma=X_hofc_train,
        add_self_loop=True,
        use_sex_gate=use_sex_gate,
        use_age_gate=use_age_gate,
        use_edu_gate=use_edu_gate,
        use_site_gate=use_site_gate,
        site_gate_mode=site_gate_mode,
        age_threshold=age_threshold,
        edu_threshold=edu_threshold,
        edu_weight=edu_weight
    )

    A_fc_norm = normalize_adj_numpy(A_fc)
    A_hofc_norm = normalize_adj_numpy(A_hofc)

    return {
        "A_fc":        numpy_to_torch_tensor(A_fc,        device=device),
        "A_hofc":      numpy_to_torch_tensor(A_hofc,      device=device),
        "A_fc_norm":   numpy_to_torch_tensor(A_fc_norm,   device=device),
        "A_hofc_norm": numpy_to_torch_tensor(A_hofc_norm, device=device),
        "sigma_fc":    sigma_fc_used,
        "sigma_hofc":  sigma_hofc_used
    }