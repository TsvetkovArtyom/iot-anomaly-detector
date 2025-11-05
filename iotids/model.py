from typing import Optional
import numpy as np, pandas as pd, os
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from .features import FeatureExtractor
from .config import TrainConfig

class ModelManager:
    def __init__(self, fx: FeatureExtractor, train_cfg: TrainConfig):
        self.fx = fx
        self.train_cfg = train_cfg
        self.scaler = StandardScaler()
        self.model = None
        self.if_shift: Optional[float] = None
        self.if_scale: Optional[float] = None

    def fit(self, df: pd.DataFrame):
        X = self.fx.transform_frame(df)
        Xs = self.scaler.fit_transform(X)
        if self.train_cfg.algo != "isolation_forest":
            raise ValueError("Only isolation_forest in this build")
        self.model = IsolationForest(
            n_estimators=self.train_cfg.n_estimators,
            contamination=self.train_cfg.contamination,
            max_samples=self.train_cfg.max_samples,
            random_state=self.train_cfg.random_state,
            n_jobs=-1,
        ).fit(Xs)
        train_scores = -self.model.score_samples(Xs)
        self.if_shift = float(np.percentile(train_scores, 1))
        self.if_scale = float(np.percentile(train_scores, 99) - self.if_shift) or 1.0

    def score(self, df: pd.DataFrame) -> np.ndarray:
        X = self.fx.transform_frame(df)
        Xs = self.scaler.transform(X)
        raw = -self.model.score_samples(Xs)  # type: ignore
        if self.if_shift is not None and self.if_scale is not None:
            return np.clip((raw - self.if_shift)/self.if_scale, 0, 1)
        p95 = np.percentile(raw, 95)
        return np.clip(raw/(p95+1e-8), 0, 1)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        dump({
            "scaler": self.scaler, "model": self.model, "train_cfg": self.train_cfg,
            "if_shift": self.if_shift, "if_scale": self.if_scale
        }, path)

    def load(self, path: str):
        b = load(path)
        self.scaler = b["scaler"]; self.model = b["model"]
        self.train_cfg = b.get("train_cfg", self.train_cfg)
        self.if_shift = b.get("if_shift"); self.if_scale = b.get("if_scale")
