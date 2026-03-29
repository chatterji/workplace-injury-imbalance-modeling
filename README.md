# Data Guidance

The original modeling dataset is confidential and is **not** included in this repository.

## Public GitHub approach

For a public portfolio repo, do not upload:
- raw employee-level records
- confidential feature values
- company-identifying fields
- internal codes or business-sensitive data exports

## Recommended workaround

Use one or more of the following:

1. **Local private data only**  
   Keep the real CSV in `data/private/pre_model_data.csv`, which is excluded by `.gitignore`.

2. **Mock data generator**  
   Use `scripts/generate_mock_data.py` to create a synthetic demonstration dataset that allows the pipeline to run publicly.

3. **Schema documentation**  
   Add a simple data dictionary describing fields at a high level without exposing confidential details.

## Minimum schema expectation

The training code expects:
- one binary target column named `CLAIM_AGG`
- numeric feature columns for model training

You can update `src/config.py` if your final public version uses a reduced or renamed feature set.
