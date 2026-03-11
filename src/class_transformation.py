from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier


class ClassTransformationUplift:
    """
    Single-model uplift estimator based on class transformation.

    The transformed target is 1 when the observed outcome aligns with a
    positive treatment effect signal and 0 otherwise.
    """

    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator or GradientBoostingClassifier(
            random_state=42
        )
        self.model = clone(self.base_estimator)

    @staticmethod
    def _transform_target(treatment: pd.Series, y: pd.Series) -> np.ndarray:
        return (treatment.to_numpy() == y.to_numpy()).astype(int)

    def fit(self, X: pd.DataFrame, treatment: pd.Series, y: pd.Series) -> "ClassTransformationUplift":
        z = self._transform_target(treatment, y)
        self.model.fit(X, z)
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.model.predict_proba(X)[:, 1]
        return 2.0 * proba - 1.0

