# src/feature_rfe.py
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


class RFESelector:
    """
    对单个视图做 RFE:
    6670 -> 500

    注意：
    1) 只能在训练集上 fit
    2) 再用同一个选择器去 transform 验证/测试集
    """

    def __init__(self, n_features_to_select=500, step=0.1, random_state=42):
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.selector = RFE(
            estimator=SVC(kernel="linear", random_state=random_state),
            n_features_to_select=n_features_to_select,
            step=step
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        X_train_std = self.scaler.fit_transform(X_train)
        self.selector.fit(X_train_std, y_train)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_std = self.scaler.transform(X)
        X_sel = self.selector.transform(X_std)
        return X_sel.astype(np.float32)

    def fit_transform(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        X_train_std = self.scaler.fit_transform(X_train)
        X_sel = self.selector.fit_transform(X_train_std, y_train)
        return X_sel.astype(np.float32)

    @property
    def support_mask(self):
        return self.selector.support_

    @property
    def ranking(self):
        return self.selector.ranking_


def apply_rfe_per_fold(
    X_fc_train, X_fc_test,
    X_hofc_train, X_hofc_test,
    y_train,
    rfe_dim=500,
    random_state=42
):
    """
    对 FC / HOFC 两个视图分别做 RFE
    返回：
        X_fc_train_sel, X_fc_test_sel,
        X_hofc_train_sel, X_hofc_test_sel,
        fc_selector, hofc_selector
    """
    fc_selector = RFESelector(
        n_features_to_select=rfe_dim,
        step=0.1,
        random_state=random_state
    )
    hofc_selector = RFESelector(
        n_features_to_select=rfe_dim,
        step=0.1,
        random_state=random_state
    )

    X_fc_train_sel = fc_selector.fit_transform(X_fc_train, y_train)
    X_fc_test_sel = fc_selector.transform(X_fc_test)

    X_hofc_train_sel = hofc_selector.fit_transform(X_hofc_train, y_train)
    X_hofc_test_sel = hofc_selector.transform(X_hofc_test)

    return (
        X_fc_train_sel, X_fc_test_sel,
        X_hofc_train_sel, X_hofc_test_sel,
        fc_selector, hofc_selector
    )


# def apply_rfe_global(
#     X_fc: np.ndarray,
#     X_hofc: np.ndarray,
#     y: np.ndarray,
#     rfe_dim: int = 500,
#     random_state: int = 42
# ):
#     """
#     先在整个数据集上做一次 RFE，得到全体样本的 500 维 FC / HOFC 特征。
#     数据泄露！故不使用此方法。
#     """
#     fc_selector = RFESelector(
#         n_features_to_select=rfe_dim,
#         step=0.1,
#         random_state=random_state
#     )
#     hofc_selector = RFESelector(
#         n_features_to_select=rfe_dim,
#         step=0.1,
#         random_state=random_state
#     )
#
#     X_fc_sel = fc_selector.fit_transform(X_fc, y)
#     X_hofc_sel = hofc_selector.fit_transform(X_hofc, y)
#
#     return X_fc_sel, X_hofc_sel, fc_selector, hofc_selector