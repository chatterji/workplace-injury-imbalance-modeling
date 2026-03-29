# Recommended Folder Structure

```text
workplace-injury-imbalance-modeling/
├── README.md                         # Employer-facing project summary
├── PROJECT_STRUCTURE.md              # Explanation of the repo layout
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Prevents private data and local artifacts from being committed
│
├── src/
│   ├── config.py                     # Centralized constants, paths, seeds, and hyperparameters
│   ├── data_prep.py                  # Data loading, splitting, scaling, and class-distribution summaries
│   ├── modeling.py                   # Model creation, training helpers, and imbalance methods
│   ├── evaluation.py                 # Metrics, plots, confusion matrices, and summary tables
│   └── train.py                      # Main training entry point
│
├── scripts/
│   └── generate_mock_data.py         # Creates synthetic/mock data for public demonstration
│
├── data/
│   └── README.md                     # Explains expected schema and why real data is excluded
│
├── notebooks/
│   └── README.md                     # Guidance for adding a polished explanatory notebook later
│
├── results/
│   └── .gitkeep                      # Placeholder for metrics, plots, and exported artifacts
│
├── docs/
│   └── publishing_without_data.md    # How to present the project without exposing confidential data
│
└── images/                           # Screenshots and exported charts for the README
```

## Why this structure works for GitHub

This layout separates:
- reusable code
- confidential-data guidance
- generated results
- narrative documentation

That makes the project easier for employers to review quickly and easier for you to maintain over time.
