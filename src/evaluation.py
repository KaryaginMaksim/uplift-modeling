from __future__ import annotations

import numpy as np
import pandas as pd


def uplift_frame(
    y_true: pd.Series,
    treatment: pd.Series,
    uplift_score: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "y": y_true.to_numpy(),
            "treatment": treatment.to_numpy(),
            "uplift_score": uplift_score,
        }
    ).sort_values("uplift_score", ascending=False)
    return frame.reset_index(drop=True)


def qini_curve(
    y_true: pd.Series,
    treatment: pd.Series,
    uplift_score: np.ndarray,
) -> pd.DataFrame:
    frame = uplift_frame(y_true, treatment, uplift_score)
    frame["treated_outcome"] = frame["y"] * (frame["treatment"] == 1)
    frame["control_outcome"] = frame["y"] * (frame["treatment"] == 0)
    frame["cum_treated"] = (frame["treatment"] == 1).cumsum()
    frame["cum_control"] = (frame["treatment"] == 0).cumsum()
    frame["cum_treated_outcome"] = frame["treated_outcome"].cumsum()
    frame["cum_control_outcome"] = frame["control_outcome"].cumsum()
    frame["incremental_gain"] = (
        frame["cum_treated_outcome"]
        - frame["cum_control_outcome"] * frame["cum_treated"] / frame["cum_control"].clip(lower=1)
    )
    frame["population_share"] = (np.arange(len(frame)) + 1) / len(frame)
    return frame[["population_share", "incremental_gain"]]


def auuc(
    y_true: pd.Series,
    treatment: pd.Series,
    uplift_score: np.ndarray,
) -> float:
    curve = qini_curve(y_true, treatment, uplift_score)
    return np.trapz(curve["incremental_gain"], curve["population_share"])

