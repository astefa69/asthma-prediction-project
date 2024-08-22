# srs/data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def handle_missing_values(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """Handle missing values in the DataFrame."""
    if strategy == 'median':
        df.fillna(df.median(), inplace=True)
    elif strategy == 'mean':
        df.fillna(df.mean(), inplace=True)
    elif strategy == 'drop':
        df.dropna(inplace=True)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return df

def encode_categorical_features(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    """Encode categorical features using one-hot encoding."""
    return pd.get_dummies(df, columns=categorical_cols, drop_first=True)

def remove_multicollinearity(df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """Remove features with high multicollinearity."""
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop)

def scale_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scale numeric features using StandardScaler."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def preprocess_data(df: pd.DataFrame, categorical_cols: list, strategy: str = 'median') -> pd.DataFrame:
    """Perform complete data preprocessing."""
    df = handle_missing_values(df, strategy)
    df = encode_categorical_features(df, categorical_cols)
    df = remove_multicollinearity(df)
    df = scale_numeric_features(df)
    return df
