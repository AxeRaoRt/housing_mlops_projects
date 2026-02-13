from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from sklearn.datasets import fetch_california_housing


@dataclass(frozen=True)
class Dataset:
    X: pd.DataFrame
    y: pd.Series
    feature_names: list[str]
    target_name: str


def load_california_housing() -> Dataset:
    data = fetch_california_housing(as_frame=True)
    X = data.data.copy()
    y = data.target.copy()
    return Dataset(X=X, y=y, feature_names=list(X.columns), target_name="MedHouseVal")
