from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _make_ohe(handle_unknown: str):
    try:
        return OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown=handle_unknown, sparse=False)


class UnknownBucketEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.categories_ = None
        self.encoder_ = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        categories = []
        for col in X.columns:
            values = pd.Series(X[col]).dropna().unique().tolist()
            if "UNK" not in values:
                values.append("UNK")
            categories.append(sorted(map(str, values)))
        self.categories_ = categories
        self.encoder_ = _make_ohe(handle_unknown="ignore")
        self.encoder_.set_params(categories=self.categories_)
        X_mapped = self._map_unknowns(X)
        self.encoder_.fit(X_mapped)
        return self

    def _map_unknowns(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for col_idx, col in enumerate(X.columns):
            known = set(self.categories_[col_idx])
            series = X_out[col].astype(str)
            series = series.where(series.isin(known), other="UNK")
            X_out[col] = series
        return X_out

    def transform(self, X):
        X = pd.DataFrame(X)
        X_mapped = self._map_unknowns(X)
        return self.encoder_.transform(X_mapped)


def build_preprocessor(schema: dict, encoding_mode: str) -> ColumnTransformer:
    num_cols: List[str] = schema["num_cols"]
    cat_cols: List[str] = schema["cat_cols"]

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    if encoding_mode == "ignore_unknown":
        cat_encoder = _make_ohe(handle_unknown="ignore")
    elif encoding_mode == "unknown_bucket":
        cat_encoder = UnknownBucketEncoder()
    else:
        raise ValueError(f"Unknown encoding_mode: {encoding_mode}")

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", cat_encoder),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return preprocessor
