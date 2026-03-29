"""Main training script for workplace injury claim prediction."""

from __future__ import annotations

from .data_prep import load_private_data, prepare_data, set_global_seeds, summarize_class_distribution
from .evaluation import evaluate_method, save_method_summary, save_training_curves
from .modeling import train_baseline_model, train_oversampled_model, train_weighted_model


def main() -> None:
    set_global_seeds()

    raw_df = load_private_data()
    class_summary = summarize_class_distribution(raw_df)

    print("Dataset summary:")
    print(f"  Total records:      {class_summary['total_count']}")
    print(f"  Positive claims:    {class_summary['positive_count']}")
    print(f"  Negative claims:    {class_summary['negative_count']}")
    print(f"  Positive rate:      {class_summary['positive_rate']:.2%}")

    prepared = prepare_data(raw_df)

    print("Prepared split rates:")
    print(f"  Train positive rate: {prepared.train_positive_rate:.2%}")
    print(f"  Validation positive rate: {prepared.val_positive_rate:.2%}")
    print(f"  Test positive rate: {prepared.test_positive_rate:.2%}")

    all_rows = []

    baseline_model, baseline_history = train_baseline_model(
        prepared.train_features,
        prepared.train_labels,
        prepared.val_features,
        prepared.val_labels,
    )
    save_training_curves(baseline_history, "baseline")
    all_rows.extend(
        evaluate_method(
            "baseline",
            baseline_model,
            prepared.test_features,
            prepared.test_labels,
            thresholds=[0.50, 0.10, 0.01],
        )
    )

    weighted_model, weighted_history = train_weighted_model(
        prepared.train_features,
        prepared.train_labels,
        prepared.val_features,
        prepared.val_labels,
    )
    save_training_curves(weighted_history, "weighted")
    all_rows.extend(
        evaluate_method(
            "class_weighted",
            weighted_model,
            prepared.test_features,
            prepared.test_labels,
            thresholds=[0.50, 0.10, 0.01],
        )
    )

    oversampled_model, oversampled_history = train_oversampled_model(
        prepared.train_features,
        prepared.train_labels,
        prepared.val_features,
        prepared.val_labels,
    )
    save_training_curves(oversampled_history, "oversampled")
    all_rows.extend(
        evaluate_method(
            "oversampled",
            oversampled_model,
            prepared.test_features,
            prepared.test_labels,
            thresholds=[0.50, 0.10, 0.01],
        )
    )

    summary_df = save_method_summary(all_rows)
    print("\nMethod comparison summary:")
    print(summary_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
