from __future__ import annotations

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier


class TwoModelUplift:
    """Estimate uplift as the difference between treatment and control models."""

    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator or RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=50,
            random_state=42,
        )
        self.treatment_model = clone(self.base_estimator)
        self.control_model = clone(self.base_estimator)

    def fit(self, X: pd.DataFrame, treatment: pd.Series, y: pd.Series) -> "TwoModelUplift":
        treat_mask = treatment == 1
        control_mask = treatment == 0

        self.treatment_model.fit(X.loc[treat_mask], y.loc[treat_mask])
        self.control_model.fit(X.loc[control_mask], y.loc[control_mask])
        return self

    def predict_uplift(self, X: pd.DataFrame):
        p_treat = self.treatment_model.predict_proba(X)[:, 1]
        p_control = self.control_model.predict_proba(X)[:, 1]
        return p_treat - p_control

