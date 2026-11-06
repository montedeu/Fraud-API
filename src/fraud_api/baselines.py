"""Baselines that show what LightGBM adds. Neither is deployed."""

import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraud_api.application import CATEGORIES, FIELDS
from fraud_api.features import prepare

Scorer = Callable[[pd.DataFrame], np.ndarray]
"""Applications in, one score per Application out, higher meaning riskier. Only LightGBM's
scores are Fraud Scores; the rule's are in ``credit_risk_score`` units."""

LOGISTIC_REGRESSION_MAX_ITER = 2000

SETTINGS = {
    "logistic_regression": {
        "numeric": "median imputation with missing indicators, standard scaling",
        "categorical": "one-hot with the fixed category lists",
        "max_iter": LOGISTIC_REGRESSION_MAX_ITER,
        "other_parameters": "scikit-learn defaults",
    },
    "credit_risk_score_rule": {"score": "credit_risk_score, higher is riskier"},
}
"""How each baseline is built, recorded next to its results."""


def fit_logistic_regression(applications: pd.DataFrame, labels: np.ndarray) -> Scorer:
    """Logistic regression with fixed settings: median imputation with missing-value
    indicators and standard scaling for numbers, one-hot encoding for categories.

    Raises if the solver does not converge, rather than reporting an unfinished fit.
    """
    numeric = [field for field in FIELDS if field not in CATEGORIES]
    pipeline = make_pipeline(
        make_column_transformer(
            (
                make_pipeline(
                    SimpleImputer(strategy="median", add_indicator=True), StandardScaler()
                ),
                numeric,
            ),
            (OneHotEncoder(categories=list(CATEGORIES.values())), list(CATEGORIES)),
        ),
        LogisticRegression(max_iter=LOGISTIC_REGRESSION_MAX_ITER),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(_as_input(applications), labels)

    def score(applications: pd.DataFrame) -> np.ndarray:
        return pipeline.predict_proba(_as_input(applications))[:, 1]

    return score


def credit_risk_score_rule(applications: pd.DataFrame) -> np.ndarray:
    """Rank Applications by ``credit_risk_score`` alone: higher is riskier.

    The scores are ``credit_risk_score`` itself, not Fraud Scores from 0 to 1. Every
    metric depends only on their order and on a Threshold chosen in the same units.
    """
    return applications["credit_risk_score"].to_numpy(dtype=float)


def _as_input(applications: pd.DataFrame) -> pd.DataFrame:
    prepared = prepare(applications, FIELDS, CATEGORIES)
    # OneHotEncoder wants plain values with its own category lists, not pandas categoricals.
    return prepared.astype({field: object for field in CATEGORIES})
