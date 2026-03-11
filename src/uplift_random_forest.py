from __future__ import annotations

import pandas as pd

try:
    from causalml.inference.tree import UpliftRandomForestClassifier
except ImportError:  # pragma: no cover
    UpliftRandomForestClassifier = None


class CausalMLUpliftRandomForest:
    """Wrapper around CausalML's uplift random forest implementation."""

    def __init__(self):
        if UpliftRandomForestClassifier is None:
            raise ImportError(
                "causalml is required for UpliftRandomForest. Install it with "
                "`pip install causalml`."
            )
        self.model = UpliftRandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=100,
            evaluationFunction="KL",
            control_name="control",
            random_state=42,
        )

    def fit(self, X: pd.DataFrame, treatment: pd.Series, y: pd.Series) -> "CausalMLUpliftRandomForest":
        treatment_labels = treatment.map({0: "control", 1: "treatment"})
        self.model.fit(X.values, treatment=treatment_labels.values, y=y.values)
        return self

    def predict_uplift(self, X: pd.DataFrame):
        predictions = self.model.predict(X.values)
        return predictions[:, 0]

