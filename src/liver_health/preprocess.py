"""
Препроцессор для ILPD:
- числовые: масштабирование и иммутация пропусков медианой
- категориальные: иммутация модой + ohe

"""

import sklearn.compose
import sklearn.pipeline
import sklearn.impute
import sklearn.preprocessing

def build_preprocessor(
        numeric_features: list[str],
        categorical_features: list[str]
) -> sklearn.compose.ColumnTransformer:
    num_pipe = sklearn.pipeline.Pipeline(steps=[
        ("imputer", sklearn.impute.SimpleImputer(strategy="median")),
        ("scaler", sklearn.preprocessing.StandardScaler()),
    ])

    # конвейер для категориальных признаков
    cat_pipe = sklearn.pipeline.Pipeline(steps=[
        ("imputer", sklearn.impute.SimpleImputer(strategy="most_frequent")),
        ("onehot", sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore")),
    ])

    # объединяем оба конвейера по колонкам
    return sklearn.compose.ColumnTransformer(transformers=[
        ("num", num_pipe, list(numeric_features)),
        ("cat", cat_pipe, list(categorical_features)),
    ])