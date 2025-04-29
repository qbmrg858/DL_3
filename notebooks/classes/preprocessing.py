# data/preprocessing.py
import numpy as np
from scipy.signal import convolve2d  # SciPy は OK、なければ自前実装でも可

class Preprocessor:
    def __init__(self, cfg):
        self.filter_on   = cfg["data"]["filter_on"]
        self.filter_type = cfg["data"].get("filter_type", "mean")  # "mean" or "gaussian"
        self.gamma_on    = cfg["data"]["gamma_on"]
        self.gamma       = cfg["data"].get("gamma", 1.0)

        # 前処理後の統計量（例：標準化用）
        self.mean_ = None
        self.std_  = None

    def fit(self, X):
        # ここでは単純に訓練データ全体の mean/std を計算
        Xf = self._filter_batch(X) if self.filter_on else X
        Xg = self._gamma_batch(Xf) if self.gamma_on else Xf
        self.mean_ = Xg.mean(axis=0)
        self.std_  = Xg.std(axis=0) + 1e-8

    def transform(self, X):
        Xf = self._filter_batch(X) if self.filter_on else X
        Xg = self._gamma_batch(Xf) if self.gamma_on else Xf
        return (Xg - self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def _filter_batch(self, X):
        # x: [N, H*W] を画像に戻してフィルタ
        N, D = X.shape
        side = int(np.sqrt(D))
        Ximg = X.reshape(N, side, side)
        out = []
        if self.filter_type == "mean":
            # 3×3 平均フィルタ
            kernel = np.ones((3,3)) / 9.0
        elif self.filter_type == "gaussian":
            # 3×3 ガウス(σ=1)
            g = np.array([[1,2,1],[2,4,2],[1,2,1]],float)
            kernel = g / g.sum()
        else:
            raise ValueError(self.filter_type)

        for img in Ximg:
            # 各チャネルがグレースケール→単一
            filtered = convolve2d(img, kernel, mode="same", boundary="symm")
            out.append(filtered.ravel())
        return np.stack(out, axis=0)

    def _gamma_batch(self, X):
        return np.power(X, self.gamma)
