"""Data preparation utilities for workplace injury claim modeling."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import (
    FEATURE_COLUMNS,
    PRIVATE_DATA_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    TF_SEED,
    VALIDATION_SIZE,
)


@dataclass
class PreparedData:
    train_features: np.ndarray
    val_features: np.ndarray
    test_features: np.ndarray
    train_labels: np.ndarray
    val_labels: np.ndarray
    test_labels: np.ndarray
    feature_names: list[str]
    scaler: StandardScaler
    train_positive_rate: float
    val_positive_rate: float
    test_positive_rate: float


def set_global_seeds() -> None:
    """Set Python, NumPy, and TensorFlow seeds for reproducible training."""
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(TF_SEED)


def load_private_data(path=PRIVATE_DATA_PATH) -> pd.DataFrame:
    """Load the private workplace injury modeling dataset from local storage."""
    return pd.read_csv(path)


def summarize_class_distribution(df: pd.DataFrame) -> dict:
    """Summarize the class balance so it can be documented in the README or logs."""
    negative_count, positive_count = np.bincount(df[TARGET_COLUMN])
    total_count = negative_count + positive_count
    positive_rate = positive_count / total_count
    return {
        "total_count": int(total_count),
        "positive_count": int(positive_count),
        "negative_count": int(negative_count),
        "positive_rate": float(positive_rate),
    }


def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate business features from the target column."""
    if FEATURE_COLUMNS:
        feature_df = df[FEATURE_COLUMNS].copy()
    else:
        feature_df = df.drop(columns=[TARGET_COLUMN]).copy()
    target_series = df[TARGET_COLUMN].copy()
    return feature_df, target_series


def prepare_data(df: pd.DataFrame) -> PreparedData:
    """Prepare train/validation/test sets and scale features.

    Why this matters:
    - The positive class is rare, so we preserve reproducibility with fixed seeds.
    - Scaling helps neural networks train more reliably on tabular data.
    - Separate validation and test sets help distinguish model tuning from final evaluation.
    """
    feature_df, target_series = split_features_and_target(df)

    train_features_df, test_features_df, train_labels, test_labels = train_test_split(
        feature_df,
        target_series,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target_series,
    )

    train_features_df, val_features_df, train_labels, val_labels = train_test_split(
        train_features_df,
        train_labels,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=train_labels,
    )

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features_df)
    val_features = scaler.transform(val_features_df)
    test_features = scaler.transform(test_features_df)

    return PreparedData(
        train_features=train_features,
        val_features=val_features,
        test_features=test_features,
        train_labels=np.array(train_labels),
        val_labels=np.array(val_labels),
        test_labels=np.array(test_labels),
        feature_names=list(train_features_df.columns),
        scaler=scaler,
        train_positive_rate=float(np.mean(train_labels)),
        val_positive_rate=float(np.mean(val_labels)),
        test_positive_rate=float(np.mean(test_labels)),
    )
