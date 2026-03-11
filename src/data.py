from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_uplift_data(
    n_samples: int = 10_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic dataset with heterogeneous treatment effects."""
    rng = np.random.default_rng(random_state)

    age = rng.integers(18, 70, n_samples)
    tenure_months = rng.integers(1, 60, n_samples)
    monthly_spend = rng.gamma(shape=3.0, scale=35.0, size=n_samples)
    sessions_30d = rng.poisson(8, n_samples)
    support_tickets_90d = rng.poisson(1.2, n_samples)
    discount_sensitivity = rng.uniform(0.0, 1.0, n_samples)

    treatment = rng.binomial(1, 0.5, n_samples)

    base_logit = (
        -2.8
        + 0.015 * age
        + 0.012 * tenure_months
        + 0.008 * monthly_spend
        + 0.09 * sessions_30d
        - 0.18 * support_tickets_90d
    )

    uplift_signal = (
        0.9 * (discount_sensitivity > 0.6)
        + 0.6 * (sessions_30d < 6)
        - 0.7 * (monthly_spend > 180)
        - 0.5 * (support_tickets_90d > 2)
    )

    logits = base_logit + treatment * uplift_signal
    prob = 1.0 / (1.0 + np.exp(-logits))
    outcome = rng.binomial(1, prob)

    return pd.DataFrame(
        {
            "age": age,
            "tenure_months": tenure_months,
            "monthly_spend": monthly_spend,
            "sessions_30d": sessions_30d,
            "support_tickets_90d": support_tickets_90d,
            "discount_sensitivity": discount_sensitivity,
            "treatment": treatment,
            "outcome": outcome,
        }
    )

