# Project Description

This project turns the original insurance EDA notebook into a runnable analytics application for premium and risk analysis.

## What It Does

- Loads the real cleaned dataset from `Data/cleaned1.csv` when it exists.
- Falls back to a synthetic insurance dataset when the DVC-tracked CSV is missing.
- Trains a regression model to predict `TotalPremium`.
- Exposes a Streamlit dashboard for:
  - dataset overview
  - province and cover-type analytics
  - premium prediction and simple risk-banding

## Why The Demo Mode Exists

The repository includes only `Data/cleaned1.csv.dvc`, not the actual `cleaned1.csv` file. That means the original notebook cannot run end to end on a fresh machine without pulling the DVC artifact first.

To keep the project runnable out of the box, this implementation supports:

- `real` mode: uses the actual cleaned CSV if present
- `demo` mode: generates a realistic synthetic dataset instantly

## Main Files

- `app.py`: Streamlit app entrypoint
- `train.py`: CLI script to train and save the model
- `src/insurance_risk/data.py`: dataset loading and synthetic data generation
- `src/insurance_risk/modeling.py`: preprocessing, model training, evaluation, and prediction
- `artifacts/`: saved model and metrics files

## Model Choice

The app uses a `RandomForestRegressor` because it is:

- easy to train quickly
- reliable on mixed tabular features
- good for a practical baseline

## Target

The current predictive target is `TotalPremium`, which makes the app useful for:

- pricing exploration
- premium estimation
- simple risk segmentation via predicted premium bands

## Next Upgrade Ideas

- add claim severity and claim probability models
- add drift checks between demo and real datasets
- add feature importance plots
- package the training pipeline with tests and CI
