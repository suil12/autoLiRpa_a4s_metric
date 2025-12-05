import numpy as np
import pandas as pd
import pytest


DATE_FEATURE = "issue_d"
N_SAMPLES: int | None = 1000


def sample(df: pd.DataFrame) -> pd.DataFrame:
    if N_SAMPLES:
        out: pd.DataFrame = df.iloc[:N_SAMPLES]
        return out

    return df


def get_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = pd.to_datetime(df[DATE_FEATURE])
    i_train = np.where(
        (pd.to_datetime("2013-01-01") <= t) & (t <= pd.to_datetime("2015-12-31"))
    )[0]
    i_test = np.where(
        (pd.to_datetime("2016-01-01") <= t) & (t <= pd.to_datetime("2017-12-31"))
    )[0]
    out: tuple[pd.DataFrame, pd.DataFrame] = df.iloc[i_train], df.iloc[i_test]
    return out


@pytest.fixture(scope="session")
def auto_load() -> None:
    auto_load("a4s_eval.metrics")


@pytest.fixture(scope="session")
def tab_class_dataset() -> pd.DataFrame:
    return pd.read_csv("./tests/data/lcld_v2.csv")


@pytest.fixture(scope="session")
def tab_class_train_data(tab_class_dataset: pd.DataFrame) -> pd.DataFrame:
    return sample(get_splits(tab_class_dataset)[0])


@pytest.fixture(scope="session")
def tab_class_test_data(tab_class_dataset: pd.DataFrame) -> pd.DataFrame:
    return sample(get_splits(tab_class_dataset)[1])
