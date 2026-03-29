# Workplace Injury Claim Prediction with Imbalanced Data

## Executive Summary

This project showcases a deep-learning solution to a workplace injury claim prediction problem with a severely imbalanced target rate of roughly **6%**. The broader analytics effort had struggled to produce an acceptable modeling approach over an extended period, largely because the minority class was both rare and business-critical. I was asked to step into a principal-level data science role and develop neural-network-based methods that could improve detection of likely injury claims without relying on accuracy alone.

The repository is structured as an employer-facing portfolio project. It demonstrates how to frame the business problem, build reproducible modeling workflows, compare multiple imbalance-handling strategies, and communicate results in terms that matter to operational leaders.

## Business Problem

Predicting workplace injury claims is challenging because positive outcomes are rare. In this project, the target class appears in only a small fraction of records. A naive model can still achieve high accuracy simply by predicting the majority class, but that approach misses the cases the business actually cares about.

For this reason, the project focuses on **minority-class detection**, not just overall accuracy. The modeling objective is to improve detection of likely injury claims while managing the tradeoff between recall and false positives.

## My Role

I was brought in specifically to address a modeling gap: the organization needed a deep-neural-network approach, and the existing team did not have that capability in place. I developed and evaluated four approaches for handling severe class imbalance in a workplace injury claim setting:

1. Baseline neural network
2. Threshold tuning
3. Class weighting
4. Oversampling for minority-class learning

## What This Project Demonstrates

* Deep learning for tabular business data
* Practical handling of severe class imbalance
* Use of **PR AUC** and **recall** instead of over-relying on accuracy
* Reproducible model training with fixed random seeds
* Modular, GitHub-ready code rather than a single experimental notebook
* Portfolio-quality project structure for prospective employers

## Repository Structure

```text
workplace-injury-imbalance-modeling/
├── README.md
├── PROJECT\_STRUCTURE.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── config.py
│   ├── data\_prep.py
│   ├── modeling.py
│   ├── evaluation.py
│   └── train.py
├── scripts/
│   └── generate\_mock\_data.py
├── data/
│   └── README.md
├── results/
│   └── .gitkeep
├── notebooks/
│   └── README.md
└── docs/
    └── publishing\_without\_data.md
```

## Modeling Approach

The original notebook explored several strategies for handling class imbalance. This repository refactors that work into reusable modules and a clearer business narrative.

### Methods Compared

**Method 1: Baseline Neural Network**  
A standard binary classifier trained on the original class distribution.

**Method 2: Threshold Tuning**  
Evaluates alternative probability thresholds to improve minority-class detection when the business cost of missed claims is high.

**Method 3: Class Weighting**  
Applies higher loss weights to the positive class so the model pays more attention to rare injury claims.

**Method 4: Oversampling**  
Rebalances the training process by oversampling positive examples to improve learning on the minority class.

## Why PR AUC and Recall Matter

In an imbalanced classification problem, accuracy can be misleading. If only a small percentage of records are positive, a model can look strong on paper while still failing to identify the events that matter most.

This project therefore emphasizes:

* **Recall**: How many true injury claims were detected
* **Precision-Recall AUC (PR AUC)**: How well the model performs across thresholds when the positive class is rare
* **Confusion-matrix tradeoffs**: Especially the balance between false negatives and false positives

That metric selection reflects real business priorities better than accuracy alone.

## Running the Project

### 1\. Install dependencies

```bash
pip install -r requirements.txt
```

### 2\. Add your private data locally

Place your confidential CSV file locally as:

```text
data/private/pre\_model\_data.csv
```

This path is intentionally ignored by Git.

### 3\. Train all methods

```bash
python -m src.train
```

### 4\. Review outputs

The training script writes artifacts to:

```text
results/
```

Expected outputs include:

* trained model metrics
* confusion-matrix summaries
* precision/recall comparisons
* method comparison table
* exported charts



## Results from the Original Notebook Run

The original notebook already contains a full model run and was used to generate the visuals and summary files included in this repository.

### Dataset imbalance

* Total observations: **111,108**
* Positive injury claims: **7,059**
* Positive rate: **6.35%**

### Summary of methods from the notebook run

|Method|Threshold|Precision|Recall|PR AUC|Key takeaway|
|-|-:|-:|-:|-:|-|
|Baseline|0.50|0.0000|0.0000|0.1334|High accuracy, but missed all positive claims|
|Baseline, threshold tuned|0.10|0.1636|0.2640|0.1334|Better detection with manageable tradeoff|
|Baseline, threshold tuned|0.01|0.0663|0.9971|0.1334|Nearly full recall, but too many false positives|
|Class weighted|0.50|0.1032|0.6373|0.1182|Strongly improved recall over baseline|
|Oversampled|0.50|0.1066|0.6702|0.1438|Best overall minority-class performance in this run|

The exported comparison file is available at:

```text
results/method\_comparison.csv
```

### Included visuals

These visuals were exported directly from the original notebook outputs and can be used in the GitHub repo README, project docs, or interview walkthroughs:

* `images/class\_distribution.png`
* `images/class\_distribution\_2.png`
* `images/baseline\_training\_metrics.png`
* `images/baseline\_confusion\_matrix\_050.png`
* `images/baseline\_confusion\_matrix\_threshold.png`
* `images/baseline\_confusion\_matrix\_threshold\_2.png`
* `images/class\_weighted\_training\_metrics.png`
* `images/class\_weighted\_confusion\_matrix\_050.png`
* `images/oversampled\_training\_metrics.png`
* `images/oversampled\_confusion\_matrix\_050.png`

### Recommended GitHub visuals to feature

For the public repo homepage, I recommend highlighting these:

1. class imbalance visual
2. baseline confusion matrix at the default threshold
3. oversampled confusion matrix
4. one training-metrics chart for the best method



## Confidential Data Strategy

This repository is designed so you can publish the project **without sharing proprietary data**.

Recommended public-GitHub approach:

* Exclude the real dataset from version control
* Include a `data/README.md` describing expected schema
* Include a mock-data generator script
* Include redacted screenshots and result summaries
* Explain the business context and methods without exposing confidential fields

## Additional Repository Files Included

This package now also includes:

* `LICENSE`
* `environment.yml`
* exported notebook visuals in `images/`
* notebook-based result summaries in `results/`
* `docs/data\_dictionary\_redacted.md`

