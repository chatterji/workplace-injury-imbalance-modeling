"""Model creation and training functions for workplace injury claim prediction."""

from __future__ import annotations

import math

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .config import (
    BATCH_SIZE,
    DROPOUT_RATE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    HIDDEN_UNITS,
)

# PR AUC and recall matter more than accuracy in this project because injury claims
# are rare. A model can appear highly accurate while still failing to identify the
# business-critical positive cases.
METRICS = [
    keras.metrics.BinaryCrossentropy(name="cross_entropy"),
    keras.metrics.MeanSquaredError(name="brier_score"),
    keras.metrics.TruePositives(name="true_positives"),
    keras.metrics.FalsePositives(name="false_positives"),
    keras.metrics.TrueNegatives(name="true_negatives"),
    keras.metrics.FalseNegatives(name="false_negatives"),
    keras.metrics.BinaryAccuracy(name="accuracy"),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
    keras.metrics.AUC(name="roc_auc"),
    keras.metrics.AUC(name="pr_auc", curve="PR"),
]


def build_injury_claim_model(input_dim: int, output_bias: float | None = None) -> keras.Model:
    """Build a simple feed-forward neural network for binary injury-claim prediction."""
    bias_initializer = None
    if output_bias is not None:
        bias_initializer = tf.keras.initializers.Constant(output_bias)

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(HIDDEN_UNITS, activation="relu"),
            keras.layers.Dropout(DROPOUT_RATE),
            keras.layers.Dense(1, activation="sigmoid", bias_initializer=bias_initializer),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=METRICS,
    )
    return model


def compute_initial_output_bias(train_labels: np.ndarray) -> float:
    """Compute an informed initial bias based on class prevalence."""
    negative_count = np.sum(train_labels == 0)
    positive_count = np.sum(train_labels == 1)
    return float(np.log(positive_count / negative_count))


def compute_class_weights(train_labels: np.ndarray) -> dict[int, float]:
    """Compute class weights so the rare injury-claim class receives more attention."""
    negative_count = np.sum(train_labels == 0)
    positive_count = np.sum(train_labels == 1)
    total_count = negative_count + positive_count
    return {
        0: (1 / negative_count) * (total_count / 2.0),
        1: (1 / positive_count) * (total_count / 2.0),
    }


def make_callbacks() -> list[keras.callbacks.Callback]:
    """Standard callbacks for stable training."""
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        )
    ]


def train_baseline_model(train_features, train_labels, val_features, val_labels) -> tuple[keras.Model, keras.callbacks.History]:
    model = build_injury_claim_model(
        input_dim=train_features.shape[1],
        output_bias=compute_initial_output_bias(train_labels),
    )
    history = model.fit(
        train_features,
        train_labels,
        validation_data=(val_features, val_labels),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=0,
        callbacks=make_callbacks(),
    )
    return model, history


def train_weighted_model(train_features, train_labels, val_features, val_labels) -> tuple[keras.Model, keras.callbacks.History]:
    model = build_injury_claim_model(
        input_dim=train_features.shape[1],
        output_bias=compute_initial_output_bias(train_labels),
    )
    history = model.fit(
        train_features,
        train_labels,
        validation_data=(val_features, val_labels),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=0,
        callbacks=make_callbacks(),
        class_weight=compute_class_weights(train_labels),
    )
    return model, history


def create_oversampled_training_set(train_features, train_labels):
    """Create a balanced tf.data training stream by oversampling the positive class."""
    positive_mask = train_labels == 1
    negative_mask = train_labels == 0

    positive_features = train_features[positive_mask]
    negative_features = train_features[negative_mask]

    positive_dataset = tf.data.Dataset.from_tensor_slices(
        (positive_features, np.ones(len(positive_features)))
    ).repeat()

    negative_dataset = tf.data.Dataset.from_tensor_slices(
        (negative_features, np.zeros(len(negative_features)))
    ).repeat()

    balanced_dataset = tf.data.Dataset.sample_from_datasets(
        [positive_dataset, negative_dataset],
        weights=[0.5, 0.5],
        seed=42,
    )

    balanced_dataset = balanced_dataset.batch(BATCH_SIZE).prefetch(2)
    steps_per_epoch = int(math.ceil(2.0 * np.sum(negative_mask) / BATCH_SIZE))
    return balanced_dataset, steps_per_epoch


def train_oversampled_model(train_features, train_labels, val_features, val_labels) -> tuple[keras.Model, keras.callbacks.History]:
    model = build_injury_claim_model(
        input_dim=train_features.shape[1],
        output_bias=compute_initial_output_bias(train_labels),
    )
    oversampled_dataset, steps_per_epoch = create_oversampled_training_set(train_features, train_labels)

    history = model.fit(
        oversampled_dataset,
        validation_data=(val_features, val_labels),
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        verbose=0,
        callbacks=make_callbacks(),
    )
    return model, history
