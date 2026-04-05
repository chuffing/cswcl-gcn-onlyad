# src/config.py
from dataclasses import dataclass
import os


@dataclass
class Config:
    # ===== 数据集 =====
    dataset_name: str = "nc_smc_lmci"

    # ===== 路径 =====
    project_root: str = "."
    data_raw_dir: str = os.path.join(project_root, "data_raw")
    runs_dir: str = os.path.join(project_root, "runs")
    ckpt_dir: str = os.path.join(runs_dir, "checkpoints")
    log_dir: str = os.path.join(runs_dir, "logs")

    # ===== 消融模式 =====
    ablation_mode: str = "cswcl"

    # ===== 训练 =====
    seed: int = 42
    n_splits: int = 5
    epochs: int = 500
    lr: float = 0.003
    weight_decay: float = 5e-4
    dropout: float = 0.3

    # ===== 特征维度 =====
    full_dim: int = 6670
    rfe_dim: int = 400

    # ===== GCN =====
    hidden_dim: int = 256
    emb_dim: int = 128
    num_gcn_layers: int = 3

    # ===== loss 权重 =====
    p_lambda_proto: float = 0.03

    lambda_cl: float = 0.01
    cp_lambda_proto: float = 0.03

    lambda_wcl: float = 0.06
    wcl_lambda_proto: float = 0.01

    lambda_cswcl: float = 0.06
    cswcl_lambda_proto: float = 0.025

    # ===== 温度 =====
    temperature: float = 0.2
    proto_temperature: float = 1.0

    # ===== Focal Loss =====
    focal_gamma: float = 2.0

    # ===== 图构建 =====
    graph_knn: int = 8
    use_sex_gate: bool = True
    use_age_gate: bool = True
    age_threshold: float = 3.0
    sigma_method: str = "median"

    # ===== 教育信息 =====
    use_edu_gate: bool = False
    edu_threshold: float = 2.0
    edu_weight: float = 0.20

    # ===== 站点 gate / 预处理 =====
    use_site_gate: bool = False
    site_gate_mode: str = "cross"
    use_fisher_z: bool = False

    # ===== Prototype query =====
    n_query: int = 1
    query_ratio: float = 0.15
    normalize_w: bool = False

    # ===== 对比学习 =====
    supervised_contrastive: bool = False

    # ===== device =====
    device: str = "cpu"

    def __post_init__(self):
        self.dataset_name = self.dataset_name.lower().strip()
        if self.dataset_name != "nc_smc_lmci":
            raise ValueError("当前版本仅支持数据集: nc_smc_lmci")

        self.ablation_mode = self.ablation_mode.lower().strip()
        valid_modes = {"mv", "p", "cp", "wcl", "cswcl"}
        if self.ablation_mode not in valid_modes:
            raise ValueError(
                f"ablation_mode 必须属于 {valid_modes}，当前得到: {self.ablation_mode}"
            )

        os.makedirs(self.runs_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
