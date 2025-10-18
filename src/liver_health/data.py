"""
Загрузка ILPD, таргет в формате +1 и -1, и три варианта валидации:
1) holdout 80/20
2) CV (5 фолдов)
3) LOOCV

"""

import sklearn.model_selection
import pandas as pd

expected_columns = [
    "Age", "Gender",
    "Total_Bilirubin", "Direct_Bilirubin",
    "Alkaline_Phosphotase",
    "Alamine_Aminotransferase", "Aspartate_Aminotransferase",
    "Total_Protiens", "Albumin", "Albumin_and_Globulin_Ratio",
    "Dataset",
]

default_numeric_features = [
    "Age", "Total_Bilirubin", "Direct_Bilirubin",
    "Alkaline_Phosphotase", "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase", "Total_Protiens",
    "Albumin", "Albumin_and_Globulin_Ratio",
]
default_categorical_features = ["Gender"]

def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    _validate_schema(df)

    return df

def _validate_schema(df: pd.DataFrame) -> None:
    missing = [col for col in expected_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Нет столбцов: {missing}. Найдены: {list(df.columns)}"
        )

def change_target(df: pd.DataFrame, drop_original: bool = True) -> pd.DataFrame:
    if 'Dataset' not in df.columns:
        raise KeyError("column 'Dataset' not found")

    out = df.copy()
    out['target'] = out['Dataset'].map({1: +1, 2: -1}).astype(int)

    if drop_original:
        out = out.drop(columns = ['Dataset'])

    return out

def split_holdout(
        df: pd.DataFrame,
        test_size = 0.2,
        random_state = 42,
):
   X = df.drop(columns = ['target'])
   y = df['target']
   return sklearn.model_selection.train_test_split(
       X, y, test_size = test_size, random_state = random_state, stratify = y
   )

def cv(
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42
) -> sklearn.model_selection.StratifiedKFold:

    return sklearn.model_selection.StratifiedKFold(
        n_splits=n_splits, shuffle=shuffle, random_state=random_state
    )


def loocv() -> sklearn.model_selection.LeaveOneOut:

    return sklearn.model_selection.LeaveOneOut()


def get_default_feature_lists() -> tuple[list[str], list[str]]:

    return default_numeric_features[:], default_categorical_features[:]


def get_X_y(df: pd.DataFrame):
    X = df.drop(columns=["target"])
    y = df["target"]

    return X, y