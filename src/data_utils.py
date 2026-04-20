# src/data_utils.py
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat


def get_upper_triangular_vector(mat: np.ndarray) -> np.ndarray:
    """
    取上三角（不含对角线，k=1）
    116x116 -> 6670
    """
    idx = np.triu_indices(mat.shape[0], k=1)
    return mat[idx].astype(np.float32)


def fc_to_hofc(fc: np.ndarray) -> np.ndarray:
    """
    按原文思路：
    将 FC 矩阵每一行视为一个脑区描述，
    再计算行与行之间的 Pearson 相关，得到 HOFC
    """
    hofc = np.corrcoef(fc)
    hofc = np.nan_to_num(hofc, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    np.fill_diagonal(hofc, 1.0)
    return hofc


def read_connectivity_mat(mat_path: str) -> np.ndarray:
    """
    读取 .mat 文件中的 connectivity 矩阵
    """
    mat = loadmat(mat_path)
    if "connectivity" not in mat:
        raise KeyError(f"{mat_path} 中没有找到键 'connectivity'")
    conn = mat["connectivity"].astype(np.float32)
    conn = np.nan_to_num(conn, nan=0.0, posinf=0.0, neginf=0.0)
    return conn


def normalize_sex_value(v):
    """
    将不同数据集里可能不同写法的 sex 统一成 0/1
    常见约定：
    - Male / M / 1 -> 1
    - Female / F / 0 / 2 -> 0
    """
    if pd.isna(v):
        return -1

    if isinstance(v, str):
        s = v.strip().lower()
        if s in ["m", "male", "man", "1"]:
            return 1
        if s in ["f", "female", "woman", "0", "2"]:
            return 0

    try:
        x = int(v)
        if x == 1:
            return 1
        if x in [0, 2]:
            return 0
    except:
        pass

    return -1


def normalize_numeric_value(v, default=-1.0):
    """
    将 AGE / EDU 这类数值型表型安全转成 float。
    缺失或非法值统一返回 default。
    """
    if pd.isna(v):
        return float(default)
    try:
        return float(v)
    except:
        return float(default)


def normalize_subject_id(v) -> str:
    """
    将 subject id 统一为目录可用字符串。
    例如 1010001.0 -> 1010001，保留原本的 sub_0001 等写法。
    """
    if pd.isna(v):
        raise ValueError("SUB_ID 存在缺失值")

    if isinstance(v, str):
        s = v.strip()
        if s.endswith(".0"):
            try:
                return str(int(float(s)))
            except:
                return s
        return s

    try:
        fv = float(v)
        if fv.is_integer():
            return str(int(fv))
    except:
        pass

    return str(v).strip()


def normalize_site_value(v) -> str:
    if pd.isna(v):
        return ""
    return normalize_subject_id(v)


def find_existing_column(df: pd.DataFrame, candidates):
    """
    在 df 中寻找候选列名，返回第一个匹配到的列名
    """
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def resolve_dataset_dir(root_dir: str, dataset_dir_name: str) -> str:
    data_dir = os.path.join(root_dir, dataset_dir_name)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"找不到数据目录: {data_dir}")

    nested_data_dir = os.path.join(data_dir, dataset_dir_name)
    if os.path.isdir(nested_data_dir):
        return nested_data_dir
    return data_dir


def find_existing_file(dir_path: str, candidates):
    for name in candidates:
        path = os.path.join(dir_path, name)
        if os.path.exists(path):
            return path
    return None


def load_dataset_from_table(data_dir: str, csv_candidates):
    csv_path = find_existing_file(data_dir, csv_candidates)
    if csv_path is None:
        raise FileNotFoundError(f"找不到表型文件，候选为: {csv_candidates}")

    pheno = pd.read_csv(csv_path)

    sub_col = find_existing_column(pheno, ["SUB_ID", "sub_id", "subject_id"])
    site_col = find_existing_column(pheno, ["SITE_ID", "site_id", "site", "Site", "SITE"])
    group_col = find_existing_column(pheno, ["Group", "group", "label", "Label"])
    sex_col = find_existing_column(pheno, ["Sex", "SEX", "sex", "Gender", "gender"])
    age_col = find_existing_column(pheno, ["Age", "AGE", "age"])
    edu_col = find_existing_column(pheno, ["EDU", "Edu", "edu", "education"])

    if sub_col is None or group_col is None:
        raise ValueError(f"{csv_path} 中未找到 SUB_ID 或 Group 列")

    X_fc, X_hofc = [], []
    y_list, sex_list, age_list, edu_list, site_list = [], [], [], [], []

    for _, row in pheno.iterrows():
        sub_id = normalize_subject_id(row[sub_col])
        mat_dir = os.path.join(data_dir, sub_id)
        mat_path = os.path.join(mat_dir, f"{sub_id}_aal_correlation.mat")

        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"找不到: {mat_path}")

        fc = read_connectivity_mat(mat_path)
        hofc = fc_to_hofc(fc)

        X_fc.append(get_upper_triangular_vector(fc))
        X_hofc.append(get_upper_triangular_vector(hofc))
        y_list.append(int(row[group_col]))

        sex_val = normalize_sex_value(row[sex_col]) if sex_col is not None else -1
        age_val = normalize_numeric_value(row[age_col]) if age_col is not None else -1.0
        edu_val = normalize_numeric_value(row[edu_col]) if edu_col is not None else -1.0
        site_val = normalize_site_value(row[site_col]) if site_col is not None else ""

        sex_list.append(sex_val)
        age_list.append(age_val)
        edu_list.append(edu_val)
        site_list.append(site_val)

    return {
        "X_fc": np.stack(X_fc, axis=0),
        "X_hofc": np.stack(X_hofc, axis=0),
        "y": np.array(y_list, dtype=np.int64),
        "sex": np.array(sex_list, dtype=np.int64),
        "age": np.array(age_list, dtype=np.float32),
        "edu": np.array(edu_list, dtype=np.float32),
        "site": np.array(site_list, dtype=object),
        "pheno": pheno
    }


def load_nc_smc_lmci(root_dir: str):
    data_dir = resolve_dataset_dir(root_dir, "NC_SMC_LMCI")
    return load_dataset_from_table(data_dir, ["NC_SMC_LMCI.csv", "phenotype.csv"])


def load_data_5(root_dir: str):
    data_dir = resolve_dataset_dir(root_dir, "data_5")
    return load_dataset_from_table(data_dir, ["phenotype.csv", "data_5.csv"])


def load_dataset(cfg):
    """
    根据配置加载不同数据集，统一返回训练所需字段。
    """
    if cfg.dataset_name == "nc_smc_lmci":
        return load_nc_smc_lmci(cfg.data_raw_dir)
    if cfg.dataset_name == "data_5":
        return load_data_5(cfg.data_raw_dir)
    raise ValueError(f"不支持的数据集: {cfg.dataset_name}")