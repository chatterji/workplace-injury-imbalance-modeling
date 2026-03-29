"""Generate mock workplace injury data for a public GitHub demo.

This script does NOT recreate the confidential dataset. It simply creates a
small synthetic dataset with a similar target imbalance so the public repo
can run end-to-end without exposing sensitive business data.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_STATE = 42
ROW_COUNT = 5000
POSITIVE_RATE = 0.06

rng = np.random.default_rng(RANDOM_STATE)

mock_df = pd.DataFrame(
    {
        "SHIFT_CASES": rng.normal(0, 1, ROW_COUNT),
        "SHIFT_KEGS": rng.normal(0, 1, ROW_COUNT),
        "TENURE_MONTHS": rng.integers(1, 240, ROW_COUNT),
        "OVERTIME_HOURS": rng.normal(10, 5, ROW_COUNT).clip(0),
        "INJURY_HISTORY_COUNT": rng.poisson(0.2, ROW_COUNT),
        "ABSENCE_DAYS": rng.poisson(2.0, ROW_COUNT),
    }
)

mock_df["CLAIM_AGG"] = (rng.random(ROW_COUNT) < POSITIVE_RATE).astype(int)

output_dir = Path(__file__).resolve().parents[1] / "data"
output_path = output_dir / "mock_pre_model_data.csv"
mock_df.to_csv(output_path, index=False)

print(f"Mock data written to {output_path}")
