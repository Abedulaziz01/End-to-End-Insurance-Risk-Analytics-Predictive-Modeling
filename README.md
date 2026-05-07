# End-to-End Insurance Risk Analytics Predictive Modeling

This repository is now a runnable insurance analytics project with:

- a fast training pipeline
- a fallback demo dataset when the DVC CSV is missing
- a Streamlit app for analytics and premium prediction

## Project Summary

The original repository focused on exploratory data analysis over South African insurance records. This version keeps that business theme and adds working application code so you can run the project quickly on a fresh machine.

The predictive target is `TotalPremium`, which is used as a pricing and risk proxy. The app supports:

- province-level premium analysis
- cover-type distribution analysis
- top vehicle-make analysis
- premium prediction for a single insurance profile
- simple risk segmentation based on predicted premium

## Project Structure

```text
.
├── app.py
├── train.py
├── PROJECT_DESCRIPTION.md
├── Data/
├── artifacts/
└── src/
    └── insurance_risk/
        ├── __init__.py
        ├── data.py
        └── modeling.py
```

## Data Behavior

This repo includes `Data/cleaned1.csv.dvc`, but the real `Data/cleaned1.csv` file is not present in the repository.

Because of that, the code runs in two modes:

1. `real` mode  
Uses `Data/cleaned1.csv` if you place the cleaned dataset there.

2. `demo` mode  
Generates a realistic synthetic insurance dataset instantly so the project always runs.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train.py
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

## Expected Outputs

Training saves:

- `artifacts/premium_model.pkl`
- `artifacts/metrics.json`

## Streamlit Features

- Overview page with dataset preview and summary metrics
- Analytics page with premium, cover, make, and claims views
- Prediction page with an insurance-profile form and instant premium estimate

## Modeling Notes

- Model: `RandomForestRegressor`
- Target: `TotalPremium`
- Metrics: MAE, RMSE, and R2

The implementation favors fast startup and practical usability over heavy experimentation.

## If You Have The Real Dataset

Place the cleaned file here:

```text
Data/cleaned1.csv
```

Then rerun:

```bash
python train.py
streamlit run app.py
```

The app will automatically switch from demo mode to real-data mode.

## Author

Original notebook analysis by Abdulaziz.  
Runnable app and pipeline structure added in this project version.
