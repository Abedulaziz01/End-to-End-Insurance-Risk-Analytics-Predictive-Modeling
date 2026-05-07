from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REAL_DATA_PATH = Path("Data/cleaned1.csv")
DEFAULT_TARGET = "TotalPremium"


@dataclass(frozen=True)
class SyntheticDataConfig:
    rows: int = 5000
    random_state: int = 42


def dataset_mode(path: Path | None = None) -> str:
    """Return whether the app is using the real dataset or demo fallback."""
    data_path = path or REAL_DATA_PATH
    return "real" if data_path.exists() else "demo"


def generate_demo_dataset(config: SyntheticDataConfig = SyntheticDataConfig()) -> pd.DataFrame:
    """Generate a fast synthetic dataset that mirrors the notebook's domain."""
    rng = np.random.default_rng(config.random_state)

    provinces = np.array(
        [
            "Gauteng",
            "KwaZulu-Natal",
            "Western Cape",
            "Eastern Cape",
            "Free State",
            "Limpopo",
            "Mpumalanga",
            "North West",
            "Northern Cape",
        ]
    )
    cover_types = np.array(
        [
            "Own Damage",
            "Third Party",
            "Comprehensive",
            "Cleaning & Debris Removal",
        ]
    )
    vehicle_types = np.array(["Sedan", "SUV", "Hatchback", "Truck", "Bus", "Van"])
    makes = np.array(
        [
            "Toyota",
            "Mercedes-Benz",
            "BMW",
            "Ford",
            "Volkswagen",
            "Isuzu",
            "Iveco",
            "Golden Journey",
            "CMC",
            "Nissan",
        ]
    )
    marital_status = np.array(["Single", "Married", "Divorced"])
    banks = np.array(["ABSA", "Standard Bank", "Nedbank", "FNB"])

    province = rng.choice(
        provinces,
        size=config.rows,
        p=np.array([0.22, 0.14, 0.13, 0.1, 0.08, 0.1, 0.08, 0.08, 0.07]),
    )
    cover_type = rng.choice(
        cover_types,
        size=config.rows,
        p=np.array([0.35, 0.3, 0.25, 0.1]),
    )
    vehicle_type = rng.choice(vehicle_types, size=config.rows)
    make = rng.choice(
        makes,
        size=config.rows,
        p=np.array([0.24, 0.12, 0.08, 0.1, 0.12, 0.07, 0.05, 0.03, 0.09, 0.1]),
    )
    marital = rng.choice(marital_status, size=config.rows, p=np.array([0.45, 0.45, 0.1]))
    bank = rng.choice(banks, size=config.rows)

    registration_year = rng.integers(2000, 2025, size=config.rows)
    vehicle_age = 2026 - registration_year
    cylinders = rng.choice([3, 4, 5, 6, 8, 10], size=config.rows, p=[0.04, 0.46, 0.08, 0.26, 0.14, 0.02])
    kilowatts = np.clip(rng.normal(95, 35, size=config.rows), 45, 260)
    sum_insured = np.clip(rng.normal(210000, 90000, size=config.rows), 50000, 950000)
    claims_frequency = rng.poisson(0.35, size=config.rows)
    total_claims = np.clip(claims_frequency * rng.normal(18000, 11000, size=config.rows), 0, None)
    capital_outstanding = np.clip(sum_insured * rng.uniform(0.15, 0.75, size=config.rows), 0, None)

    province_risk = {
        "Gauteng": 14.0,
        "KwaZulu-Natal": 18.0,
        "Western Cape": 12.5,
        "Eastern Cape": 10.0,
        "Free State": 7.5,
        "Limpopo": 8.5,
        "Mpumalanga": 11.0,
        "North West": 7.0,
        "Northern Cape": 5.5,
    }
    cover_risk = {
        "Own Damage": 19.0,
        "Third Party": 8.5,
        "Comprehensive": 22.0,
        "Cleaning & Debris Removal": 6.0,
    }
    vehicle_risk = {
        "Sedan": 4.0,
        "SUV": 6.0,
        "Hatchback": 3.0,
        "Truck": 10.0,
        "Bus": 11.0,
        "Van": 5.0,
    }
    make_risk = {
        "Toyota": 5.0,
        "Mercedes-Benz": 11.0,
        "BMW": 12.0,
        "Ford": 6.0,
        "Volkswagen": 6.5,
        "Isuzu": 7.5,
        "Iveco": 9.5,
        "Golden Journey": 10.5,
        "CMC": 4.5,
        "Nissan": 6.0,
    }

    premium = (
        18
        + 0.00016 * sum_insured
        + 0.00022 * capital_outstanding
        + 0.07 * kilowatts
        + 0.95 * vehicle_age
        + 0.00055 * total_claims
        + np.vectorize(province_risk.get)(province)
        + np.vectorize(cover_risk.get)(cover_type)
        + np.vectorize(vehicle_risk.get)(vehicle_type)
        + np.vectorize(make_risk.get)(make)
        + rng.normal(0, 9, size=config.rows)
    )
    premium = np.clip(premium, 18, None)
    calculated_premium = premium * rng.uniform(0.92, 1.08, size=config.rows)

    df = pd.DataFrame(
        {
            "Province": province,
            "CoverType": cover_type,
            "VehicleType": vehicle_type,
            "make": make,
            "MaritalStatus": marital,
            "Bank": bank,
            "RegistrationYear": registration_year,
            "VehicleAge": vehicle_age,
            "Cylinders": cylinders,
            "kilowatts": np.round(kilowatts, 2),
            "SumInsured": np.round(sum_insured, 2),
            "CapitalOutstanding": np.round(capital_outstanding, 2),
            "CalculatedPremiumPerTerm": np.round(calculated_premium, 2),
            "TotalClaims": np.round(total_claims, 2),
            "ClaimCount": claims_frequency,
            "TotalPremium": np.round(premium, 2),
        }
    )
    return df


def clean_dataset(df: pd.DataFrame, target: str = DEFAULT_TARGET) -> pd.DataFrame:
    """Keep the dataset consistent across the real and synthetic modes."""
    cleaned = df.copy()

    if target not in cleaned.columns:
        raise ValueError(f"Expected target column '{target}' in dataset.")

    cleaned = cleaned.dropna(subset=[target]).drop_duplicates()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]

    for column in cleaned.columns:
        if cleaned[column].dtype == "object":
            cleaned[column] = cleaned[column].fillna("Unknown")
        elif column != target:
            cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    if "VehicleAge" not in cleaned.columns and "RegistrationYear" in cleaned.columns:
        cleaned["VehicleAge"] = 2026 - cleaned["RegistrationYear"]

    return cleaned


def load_dataset(
    path: Path | None = None,
    target: str = DEFAULT_TARGET,
    demo_rows: int = 5000,
    max_rows: int | None = 50000,
) -> tuple[pd.DataFrame, str]:
    """Load the real dataset when available, otherwise fall back to demo data."""
    data_path = path or REAL_DATA_PATH
    if data_path.exists():
        df = pd.read_csv(data_path)
        if max_rows:
            df = df.head(max_rows)
        return clean_dataset(df, target=target), "real"

    demo_df = generate_demo_dataset(SyntheticDataConfig(rows=demo_rows))
    return clean_dataset(demo_df, target=target), "demo"
