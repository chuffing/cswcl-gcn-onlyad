# src/config.py
from dataclasses import dataclass
import os
import torch


SUPPORTED_DATASETS = {"nc_smc_lmci", "data_5"}
VALID_ABLATION_MODES = {"mv", "p", "cp", "wcl", "cswcl"}


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
    use_knn: bool = True
    graph_knn: int = 8
    use_sex_gate: bool = True
    use_age_gate: bool = True
    age_threshold: float = 3.0
    sigma_method: str = "median"

    # ===== 教育信息 =====
    use_edu_gate: bool = False  # 原论文没用我也不用了而且加上去没效果
    edu_threshold: float = 2.0
    edu_weight: float = 0.20

    # ===== 站点 gate / 预处理 =====
    use_site_gate: bool = False
    site_gate_mode: str = "cross"
    use_fisher_z: bool = False

    # ===== Prototype query =====
    n_query: int = 1
    query_ratio: float = 0.1
    proto_resample_interval: int = 1
    normalize_w: bool = False

    # ===== 对比学习 =====
    supervised_contrastive: bool = False

    # ===== device =====
    device: str = "cpu"

    def _apply_data5_defaults(self):
        # 仅覆盖当前代码里实际会用到的配置项，避免引入未接线参数。
        self.ablation_mode = "mv"
        self.seed = 42
        self.n_splits = 5
        self.epochs = 500
        self.lr = 0.003
        self.weight_decay = 5e-4
        self.dropout = 0.3

        self.rfe_dim = 500
        self.hidden_dim = 256
        self.emb_dim = 128

        self.p_lambda_proto = 0.04

        self.lambda_cl = 0.01
        self.cp_lambda_proto = 0.05

        self.lambda_wcl = 0.02
        self.wcl_lambda_proto = 0.04

        self.lambda_cswcl = 0.03  # 0.02
        self.cswcl_lambda_proto = 0.05

        self.temperature = 0.5
        self.proto_temperature = 1.0
        self.focal_gamma = 2.0

        self.graph_knn = 10
        self.use_sex_gate = False
        self.use_age_gate = False
        self.age_threshold = 15.0
        self.sigma_method = "percentile25"

        self.use_site_gate = True
        self.site_gate_mode = "cross"
        self.proto_resample_interval = 20
        self.device = "cuda"

    def _resolve_device(self):
        self.device = str(self.device).lower().strip()
        if self.device == "cuda" and not torch.cuda.is_available():
            print("警告: 当前 PyTorch 不支持 CUDA，device 已自动回退为 cpu。")
            self.device = "cpu"

    def __post_init__(self):
        self.dataset_name = self.dataset_name.lower().strip()
        if self.dataset_name not in SUPPORTED_DATASETS:
            raise ValueError(f"当前版本仅支持数据集: {SUPPORTED_DATASETS}")

        if self.dataset_name == "data_5":
            self._apply_data5_defaults()

        self.ablation_mode = self.ablation_mode.lower().strip()
        if self.ablation_mode not in VALID_ABLATION_MODES:
            raise ValueError(
                f"ablation_mode 必须属于 {VALID_ABLATION_MODES}，当前得到: {self.ablation_mode}"
            )
        if self.n_query is not None and self.n_query < 1:
            raise ValueError("n_query 必须为正整数或 None")
        if not (0.0 < self.query_ratio < 1.0):
            raise ValueError("query_ratio 必须在 (0, 1) 区间内")
        if self.proto_resample_interval < 1:
            raise ValueError("proto_resample_interval 必须 >= 1")
        if self.use_knn and self.graph_knn < 1:
            raise ValueError("开启 kNN 时，graph_knn 必须 >= 1")

        self._resolve_device()

        os.makedirs(self.runs_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
