"""Turn Applications into model input. Training and serving both go through here."""

from collections.abc import Mapping, Sequence

import pandas as pd


def prepare(
    applications: pd.DataFrame,
    features: Sequence[str],
    categories: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Model input for ``applications``: one column per feature, in ``features`` order.

    Missing values become NaN. Category fields use ``categories`` as their fixed
    category lists, so the same value always gets the same code, whichever
    Applications are prepared together.
    """
    if missing := set(features) - set(applications.columns):
        raise ValueError(f"Applications are missing fields: {sorted(missing)}")
    columns = {}
    for field in features:
        values = applications[field]
        if field in categories:
            allowed = list(categories[field])
            if unknown := set(values.dropna()) - set(allowed):
                raise ValueError(f"{field} has unknown categories: {sorted(unknown)}")
            columns[field] = pd.Categorical(values, categories=allowed)
        else:
            columns[field] = pd.to_numeric(values).astype("float64")
    return pd.DataFrame(columns, index=applications.index)
