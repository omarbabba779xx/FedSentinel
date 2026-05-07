"""
Feature engineering and preprocessing for NSL-KDD and CICIDS2017.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from utils.logger import get_logger

logger = get_logger("Preprocessor")

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
LABEL_COLS = ["label", "label_num", "difficulty"]


class NSLKDDPreprocessor:
    def __init__(self, scaler_type: str = "standard", feature_selection: bool = False, k_features: int = 50):
        self.scaler_type = scaler_type
        self.feature_selection = feature_selection
        self.k_features = k_features
        self.scaler = None
        self.label_encoders: dict = {}
        self.feature_selector = None
        self.feature_names: List[str] = []
        self.is_fitted = False

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        df = df.copy()
        for col in CATEGORICAL_COLS:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le is None:
                    df[col] = 0
                    continue
                known = set(le.classes_)
                df[col] = df[col].astype(str).apply(lambda x: x if x in known else le.classes_[0])
                df[col] = le.transform(df[col])
        return df

    def _drop_non_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        drop_cols = [c for c in LABEL_COLS if c in df.columns]
        y = df["label_num"].values if "label_num" in df.columns else np.zeros(len(df))
        X = df.drop(columns=drop_cols, errors="ignore")
        return X, y

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        df = self._encode_categoricals(df, fit=True)
        X, y = self._drop_non_features(df)

        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
        self.feature_names = list(X.columns)

        if self.scaler_type == "standard":
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()

        X_scaled = self.scaler.fit_transform(X.values)

        if self.feature_selection:
            self.feature_selector = SelectKBest(f_classif, k=min(self.k_features, X_scaled.shape[1]))
            X_scaled = self.feature_selector.fit_transform(X_scaled, y)
            logger.info(f"Feature selection: {X.shape[1]} → {X_scaled.shape[1]} features")

        self.is_fitted = True
        logger.info(f"Fitted preprocessor on {len(df)} samples, {X_scaled.shape[1]} features")
        return X_scaled.astype(np.float32), y.astype(np.int64)

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform first.")
        df = self._encode_categoricals(df, fit=False)
        X, y = self._drop_non_features(df)

        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

        X_scaled = self.scaler.transform(X.values)

        if self.feature_selection and self.feature_selector is not None:
            X_scaled = self.feature_selector.transform(X_scaled)

        return X_scaled.astype(np.float32), y.astype(np.int64)

    def save(self, path: str = "./data/preprocessor.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Preprocessor saved to {path}")

    @classmethod
    def load(cls, path: str = "./data/preprocessor.pkl") -> "NSLKDDPreprocessor":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Preprocessor loaded from {path}")
        return obj

    @property
    def num_features(self) -> int:
        if self.feature_selection and self.feature_selector is not None:
            return self.k_features
        return len(self.feature_names) if self.feature_names else 122
