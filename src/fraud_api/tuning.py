"""Hyperparameter search with Optuna on the Tuning Month (ADR 0001).

Every trial trains on Months 0–3 and is scored by Recall@5%FPR on Month 4, so neither
the Validation Month nor the Test Months can influence the chosen configuration.
"""

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from fraud_api import evaluation
from fraud_api.application import CATEGORIES, FIELDS, Dataset
from fraud_api.features import prepare

TUNING_TRAINING_MONTHS = (0, 1, 2, 3)
TUNING_MONTH = 4
N_TRIALS = 50
MAX_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 50
DEFAULT_BOOST_ROUND = 100  # LightGBM's default


@dataclass(frozen=True)
class TuningResult:
    params: dict[str, Any]
    """Best trial's search-space parameters, on top of the fixed ones."""
    num_boost_round: int
    """Best trial's early-stopped tree count, reused without early stopping later."""
    default_score: float
    """Recall@5%FPR on the Tuning Month with LightGBM's default configuration."""
    tuned_score: float
    """Recall@5%FPR on the Tuning Month of the best trial."""
    study: optuna.Study


def fixed_params(seed: int) -> dict[str, Any]:
    """Parameters every LightGBM fit shares, tuned or not."""
    return {
        "objective": "binary",
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "verbosity": -1,
    }


def search_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 500, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": 1,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 20.0),
    }


def tune(dataset: Dataset, *, n_trials: int = N_TRIALS, seed: int) -> TuningResult:
    """Search LightGBM configurations using only the Tuning Months of ``dataset``."""
    train_mask = np.isin(dataset.months, TUNING_TRAINING_MONTHS)
    tuning_mask = dataset.months == TUNING_MONTH
    x_train = prepare(dataset.applications[train_mask], FIELDS, CATEGORIES)
    y_train = dataset.labels[train_mask]
    x_tune = prepare(dataset.applications[tuning_mask], FIELDS, CATEGORIES)
    y_tune = dataset.labels[tuning_mask]

    def recall(booster: lgb.Booster, num_iteration: int | None = None) -> float:
        scores = booster.predict(x_tune, num_iteration=num_iteration)
        return evaluation.recall_at_fpr(y_tune, scores)

    default = lgb.train(
        fixed_params(seed), lgb.Dataset(x_train, y_train), num_boost_round=DEFAULT_BOOST_ROUND
    )
    default_score = recall(default)

    def objective(trial: optuna.Trial) -> float:
        train_set = lgb.Dataset(x_train, y_train)
        booster = lgb.train(
            fixed_params(seed) | search_space(trial) | {"metric": "auc"},
            train_set,
            num_boost_round=MAX_BOOST_ROUND,
            valid_sets=[lgb.Dataset(x_tune, y_tune, reference=train_set)],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        trial.set_user_attr("num_boost_round", booster.best_iteration)
        return recall(booster, booster.best_iteration)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name="lightgbm-tuning-month",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    return TuningResult(
        params=fixed_params(seed) | search_space(optuna.trial.FixedTrial(best.params)),
        num_boost_round=best.user_attrs["num_boost_round"],
        default_score=default_score,
        tuned_score=best.value,
        study=study,
    )


def trials_table(study: optuna.Study) -> pd.DataFrame:
    """One row per trial: its parameters, tree count and Tuning Month score."""
    table = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    return table.rename(columns={"value": "tuning_month_recall_at_5pct_fpr"})


def param_importances(study: optuna.Study, seed: int) -> dict[str, float]:
    evaluator = optuna.importance.FanovaImportanceEvaluator(seed=seed)
    return optuna.importance.get_param_importances(study, evaluator=evaluator)
