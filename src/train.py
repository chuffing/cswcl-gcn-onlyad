# src/train.py
import random
import numpy as np
import torch
import torch.optim as optim

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize

from src.data_utils import load_dataset
from src.feature_rfe import apply_rfe_per_fold
from src.graph_utils import build_two_view_graphs
from src.model import CSWCLGCN
from src.losses_builder import total_loss
from src.losses.prototype_loss import SharedDistanceClassifier


import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true, y_pred, y_prob, num_classes):
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    bacc     = balanced_accuracy_score(y_true, y_pred)
    try:
        if num_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            y_bin = label_binarize(y_true, classes=np.arange(num_classes))
            auc   = roc_auc_score(y_bin, y_prob, multi_class="ovo", average="macro")
    except ValueError:
        auc = 0.0
    return f1_macro, f1_weighted, bacc, auc


def mode_to_model_name(mode: str) -> str:
    mapping = {
        "mv": "MV-GCN",
        "p": "P-GCN",
        "cp": "CP-GCN",
        "wcl": "WCL-GCN",
        "cswcl": "CSWCL-GCN",
    }
    mode = mode.lower().strip()
    if mode not in mapping:
        raise ValueError(f"未知 ablation mode: {mode}")
    return mapping[mode]


def run_5fold_training(cfg):
    set_seed(cfg.seed)

    mode = cfg.ablation_mode.lower().strip()
    model_name = mode_to_model_name(mode)

    data = load_dataset(cfg)
    X_fc   = data["X_fc"]
    X_hofc = data["X_hofc"]
    y      = data["y"]
    sex    = data["sex"]
    age    = data["age"]
    edu = data["edu"]
    site = data["site"]

    num_classes = len(np.unique(y))
    print(f"数据集: {cfg.dataset_name} | 模式: {mode} | 模型: {model_name}")
    print(f"类别数: {num_classes} | 样本数: {len(y)}")
    print(f"类别分布: {{ {', '.join([f'{k}: {int((y == k).sum())}' for k in range(num_classes)])} }}")

    skf = StratifiedKFold(
        n_splits=cfg.n_splits,
        shuffle=True,
        random_state=cfg.seed
    )

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_fc, y), start=1):
        print(f"\n================ Fold {fold}/{cfg.n_splits} ================")

        # 1) 划分
        X_fc_train,   X_fc_test   = X_fc[train_idx],   X_fc[test_idx]
        X_hofc_train, X_hofc_test = X_hofc[train_idx], X_hofc[test_idx]
        y_train,      y_test      = y[train_idx],      y[test_idx]
        sex_train,    sex_test    = sex[train_idx],    sex[test_idx]
        age_train, age_test = age[train_idx], age[test_idx]
        edu_train, edu_test = edu[train_idx], edu[test_idx]
        site_train, site_test = site[train_idx], site[test_idx]

        # 2) RFE（只在训练集 fit）
        (
            X_fc_train_sel, X_fc_test_sel,
            X_hofc_train_sel, X_hofc_test_sel,
            _, _   # 不关注选择器本身
        ) = apply_rfe_per_fold(
            X_fc_train, X_fc_test,
            X_hofc_train, X_hofc_test,
            y_train,
            rfe_dim=cfg.rfe_dim,
            random_state=cfg.seed + fold
        )
        print(f"RFE完成 | FC: {X_fc_train_sel.shape} | HOFC: {X_hofc_train_sel.shape}")

        # 3) 拼成折内全图（前=train，后=test）
        X_fc_fold   = np.concatenate([X_fc_train_sel,   X_fc_test_sel],   axis=0)
        X_hofc_fold = np.concatenate([X_hofc_train_sel, X_hofc_test_sel], axis=0)
        y_fold      = np.concatenate([y_train, y_test], axis=0)
        sex_fold    = np.concatenate([sex_train, sex_test], axis=0)
        age_fold = np.concatenate([age_train, age_test], axis=0)
        edu_fold = np.concatenate([edu_train, edu_test], axis=0)
        site_fold = np.concatenate([site_train, site_test], axis=0)
        graph_knn = cfg.graph_knn if cfg.use_knn else None

        # 4) 构图
        graphs = build_two_view_graphs(
            X_fc=X_fc_fold,
            X_hofc=X_hofc_fold,
            sex=sex_fold,
            age=age_fold,
            edu=edu_fold,
            site=site_fold,
            sigma_fc=None,
            sigma_hofc=None,
            sigma_method=cfg.sigma_method,
            knn=graph_knn,
            X_fc_train=X_fc_train_sel,
            X_hofc_train=X_hofc_train_sel,
            use_sex_gate=cfg.use_sex_gate,
            use_age_gate=cfg.use_age_gate,
            use_edu_gate=cfg.use_edu_gate,
            use_site_gate=cfg.use_site_gate,
            site_gate_mode=cfg.site_gate_mode,
            age_threshold=cfg.age_threshold,
            edu_threshold=cfg.edu_threshold,
            edu_weight=cfg.edu_weight,
            device=cfg.device
        )

        print(
            f"图构建完成 | sigma_fc={graphs['sigma_fc']:.2f} | "
            f"sigma_hofc={graphs['sigma_hofc']:.2f} | "
            f"knn={'off' if graph_knn is None else graph_knn}"
        )

        # 5) 转 tensor
        x_fc_t   = torch.tensor(X_fc_fold,   dtype=torch.float32, device=cfg.device)
        x_hofc_t = torch.tensor(X_hofc_fold, dtype=torch.float32, device=cfg.device)
        y_all_t  = torch.tensor(y_fold,      dtype=torch.long,    device=cfg.device)

        n_train     = len(train_idx)  # 当前折训练样本数
        n_test      = len(test_idx)  # 当前折测试样本数
        # 下为索引，用于在全图中分割出训练集和测试集
        train_idx_t = torch.arange(n_train, dtype=torch.long, device=cfg.device)
        test_idx_t  = torch.arange(n_train, n_train + n_test, dtype=torch.long, device=cfg.device)

        # 6) 建模
        model = CSWCLGCN(
            in_dim=cfg.rfe_dim,
            hidden_dim=cfg.hidden_dim,
            emb_dim=cfg.emb_dim,
            num_classes=num_classes,
            dropout=cfg.dropout
        ).to(cfg.device)

        proto_classifier = SharedDistanceClassifier(
            temperature=cfg.proto_temperature
        ).to(cfg.device)

        optimizer = optim.Adam(
            list(model.parameters()) + list(proto_classifier.parameters()),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )

        # 7) 训练
        best_score = -1.0
        best_epoch = 0
        best_state = None
        best_metrics = None

        for epoch in range(1, cfg.epochs + 1):
            proto_cycle = (epoch - 1) // cfg.proto_resample_interval
            proto_seed = cfg.seed + fold * 1000 + proto_cycle
            model.train()
            proto_classifier.train()
            optimizer.zero_grad()

            outputs = model(
                x_fc_t,
                x_hofc_t,
                graphs["A_fc_norm"],
                graphs["A_hofc_norm"]
            )

            # 训练损失只在训练样本上算
            train_outputs = {
                "logits": outputs["logits"][train_idx_t],
                "z_fc":   outputs["z_fc"][train_idx_t],
                "z_hofc": outputs["z_hofc"][train_idx_t],
            }
            y_train_t = y_all_t[train_idx_t]

            loss, loss_dict = total_loss(
                outputs=train_outputs,
                y=y_train_t,
                num_classes=num_classes,
                proto_classifier=proto_classifier,
                n_query=cfg.n_query,
                query_ratio=cfg.query_ratio,
                mode=mode,
                # p
                p_lambda_proto=cfg.p_lambda_proto,

                # cp
                lambda_cl=cfg.lambda_cl,
                cp_lambda_proto=cfg.cp_lambda_proto,

                # wcl
                lambda_wcl=cfg.lambda_wcl,
                wcl_lambda_proto=cfg.wcl_lambda_proto,

                # cswcl
                lambda_cswcl=cfg.lambda_cswcl,
                cswcl_lambda_proto=cfg.cswcl_lambda_proto,

                # shared
                temperature=cfg.temperature,
                focal_gamma=cfg.focal_gamma,
                seed=proto_seed
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(proto_classifier.parameters()),
                max_norm=1.0
            )
            optimizer.step()

            # 测试集评估（作为 early stopping 依据）
            model.eval()
            with torch.no_grad():
                out_eval = model(
                    x_fc_t,
                    x_hofc_t,
                    graphs["A_fc_norm"],
                    graphs["A_hofc_norm"]
                )
                logits_test = out_eval["logits"][test_idx_t]
                prob = torch.softmax(logits_test, dim=1).cpu().numpy()
                pred = np.argmax(prob, axis=1)
                y_np = y_fold[test_idx_t.cpu().numpy()]
                f1_macro, f1_weighted, bacc, auc = compute_metrics(y_np, pred, prob, num_classes)
                score = 0.4 * f1_macro + 0.2 * f1_weighted + 0.25 * bacc + 0.15 * auc

            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = {
                    "model": {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    },
                    "proto_classifier": {
                        k: v.detach().cpu().clone()
                        for k, v in proto_classifier.state_dict().items()
                    }
                }
                best_metrics = {
                    "f1_macro": f1_macro,
                    "f1_weighted": f1_weighted,
                    "bacc": bacc,
                    "auc": auc,
                    "score": score
                }

            if epoch % 50 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:03d} | "
                    f"loss={loss.item():.4f} | "
                    f"cls={float(loss_dict['l_class']):.3f} | "
                    f"proto={float(loss_dict['l_proto']):.3f} | "
                    f"cl={float(loss_dict['l_cl']):.3f} | "
                    f"wcl={float(loss_dict['l_wcl']):.3f} | "
                    f"cswcl={float(loss_dict['l_cswcl']):.3f} | "
                    f"F1={f1_macro*100:.1f} F1w={f1_weighted*100:.1f} "
                    f"BACC={bacc*100:.1f} AUC={auc*100:.1f}"
                )

        # 8) 加载最佳模型，最终测试
        model.load_state_dict(best_state["model"])
        proto_classifier.load_state_dict(best_state["proto_classifier"])

        model.eval()
        with torch.no_grad():
            out_eval = model(
                x_fc_t,
                x_hofc_t,
                graphs["A_fc_norm"],
                graphs["A_hofc_norm"]
            )
            logits_test = out_eval["logits"][test_idx_t]
            prob = torch.softmax(logits_test, dim=1).cpu().numpy()
            pred = np.argmax(prob, axis=1)
            y_np = y_fold[test_idx_t.cpu().numpy()]
            f1_macro, f1_weighted, bacc, auc = compute_metrics(y_np, pred, prob, num_classes)

        print(f"\n>>> Fold {fold} 最佳epoch: {best_epoch}")
        print(
            f">>> Fold {fold} | Score={best_metrics['score'] * 100:.2f} | "
            f"F1-macro={f1_macro * 100:.2f} | "
            f"F1-weighted={f1_weighted * 100:.2f} | "
            f"BACC={bacc * 100:.2f} | AUC={auc * 100:.2f}"
        )

        fold_results.append({
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "bacc": bacc,
            "auc": auc
        })

    # 9) 汇总
    f1s   = np.array([r["f1_macro"] for r in fold_results]) * 100
    f1ws  = np.array([r["f1_weighted"] for r in fold_results]) * 100
    baccs = np.array([r["bacc"]     for r in fold_results]) * 100
    aucs  = np.array([r["auc"]      for r in fold_results]) * 100

    print("\n================= FINAL (mean ± std) =================")
    print(f"Mode: {mode} | Data: {cfg.dataset_name}")
    print(f"Model: {model_name}")
    print(f"F1-macro : {f1s.mean():.2f} ± {f1s.std():.2f}")
    print(f"F1-weighted : {f1ws.mean():.2f} ± {f1ws.std():.2f}")
    print(f"B-ACC    : {baccs.mean():.2f} ± {baccs.std():.2f}")
    print(f"AUC      : {aucs.mean():.2f} ± {aucs.std():.2f}")

    return {
        "mode": mode,
        "model_name": model_name,
        "f1_mean":   f1s.mean(),
        "f1_std":    f1s.std(),
        "f1_weighted_mean": f1ws.mean(),
        "f1_weighted_std":  f1ws.std(),
        "bacc_mean": baccs.mean(),
        "bacc_std":  baccs.std(),
        "auc_mean":  aucs.mean(),
        "auc_std":   aucs.std(),
        "fold_results": fold_results
    }