import pandas as pd


def cement_plus_water_feature(df: pd.DataFrame) -> pd.Series:
    return df["cement"] + df["water"]


PROMOTED_EXPERIMENTAL_FEATURES = [
    {
        "name": "cement_plus_water_feature",
        "function": cement_plus_water_feature,
        "source": "tests",
    }
]
