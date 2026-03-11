from __future__ import annotations

from sklearn.model_selection import train_test_split

from data import make_synthetic_uplift_data
from evaluation import auuc
from two_model import TwoModelUplift
from class_transformation import ClassTransformationUplift


def main() -> None:
    df = make_synthetic_uplift_data()
    X = df.drop(columns=["treatment", "outcome"])
    treatment = df["treatment"]
    y = df["outcome"]

    X_train, X_test, t_train, t_test, y_train, y_test = train_test_split(
        X, treatment, y, test_size=0.3, random_state=42
    )

    two_model = TwoModelUplift().fit(X_train, t_train, y_train)
    ct_model = ClassTransformationUplift().fit(X_train, t_train, y_train)

    two_model_auuc = auuc(y_test, t_test, two_model.predict_uplift(X_test))
    ct_model_auuc = auuc(y_test, t_test, ct_model.predict_uplift(X_test))

    print("Two-Model AUUC:", round(two_model_auuc, 4))
    print("Class Transformation AUUC:", round(ct_model_auuc, 4))
    print("Uplift Random Forest is available via src/uplift_random_forest.py")


if __name__ == "__main__":
    main()
