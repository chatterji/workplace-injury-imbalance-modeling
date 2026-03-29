# How to Publish This Project Without Sharing Confidential Data

The best public-GitHub strategy is to publish the **problem framing, methodology, code structure, and mock execution path** without publishing the real data.

## Recommended setup

### Option 1: Private data stays local
- Keep the actual dataset only on your machine
- Store it in `data/private/pre_model_data.csv`
- Exclude that folder with `.gitignore`

### Option 2: Public mock dataset
- Use the included mock-data generator
- Show that the code runs end-to-end on synthetic data
- Explain that the production project used confidential workplace safety data

### Option 3: Screenshots + result tables
- Export selected plots from your private run
- Redact any sensitive feature names if needed
- Include:
  - class-imbalance chart
  - confusion matrices
  - final model comparison table

## What employers care about most

Employers generally do not need the real data to evaluate your work. They want to see:
- your problem framing
- your modeling judgment
- your ability to handle class imbalance
- your code quality
- your ability to communicate results clearly

## Recommended public statement

You can use wording like this:

> Due to data confidentiality, the original dataset is not included. This repository contains a refactored version of the modeling workflow, a mock-data generator for public demonstration, and documentation of the methods used to address severe class imbalance in a workplace injury claim prediction setting.
