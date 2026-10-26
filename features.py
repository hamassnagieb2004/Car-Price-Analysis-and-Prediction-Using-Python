"""Feature engineering for the car-price model (raw DataFrame -> model-ready DataFrame).

Everything here is row-wise (no statistics learned from data), so it is safe to run
before the train/test split with no leakage. Learned steps (encoding, imputation)
live inside the sklearn Pipeline in train.py.
"""
import re
import numpy as np
import pandas as pd

CURRENT_YEAR = 2026

NUM_COLS = [
    "car_age", "km_per_year", "log_km", "engine_cc", "power_bhp", "torque_nm",
    "power_per_cc", "Fuel Tank Capacity", "Seating Capacity", "Volume",
    "Length", "Width", "Height",
]
HIGH_CARD_COLS = ["Make", "Model_clean", "Make_Fuel", "Location"]       # target-encoded
LOW_CARD_COLS = ["Fuel Type", "Transmission", "Owner", "Seller Type", "Drivetrain", "Color"]


def clean_model(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.lower()
    s = s.str.replace(r"\s*\[\s*\d{4}(?:\s*-\s*\d{4})?\s*\]", "", regex=True)
    s = s.str.replace(r"[‐-‒–—]", "-", regex=True)
    s = s.str.replace(r"\s*\+\s*", " + ", regex=True)
    s = s.str.replace(r"\b(xdrive|sdrive)\s+(\d)", r"\1\2", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def base_model(make: pd.Series, model_clean: pd.Series) -> pd.Series:
    """'Make + first word(s) of model' e.g. 'bmw x1', 'maruti suzuki swift'.
    Gives the encoder a mid-level grouping, so rare trims borrow strength from siblings."""
    first = model_clean.fillna("").str.split().str[0].fillna("")
    return (make.astype("string").str.lower() + " " + first).str.strip()


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    X["car_age"] = (CURRENT_YEAR - X["Year"]).clip(lower=0)
    X["km_per_year"] = X["Kilometer"].clip(upper=500_000) / X["car_age"].clip(lower=1)
    X["log_km"] = np.log1p(X["Kilometer"].clip(upper=500_000))

    X["engine_cc"] = pd.to_numeric(
        X["Engine"].astype("string").str.extract(r"(\d+(?:\.\d+)?)\s*cc", flags=re.I)[0],
        errors="coerce")
    X["power_bhp"] = pd.to_numeric(
        X["Max Power"].astype("string").str.extract(r"^\s*([\d.]+)")[0], errors="coerce")
    X["torque_nm"] = pd.to_numeric(
        X["Max Torque"].astype("string").str.extract(r"^\s*([\d.]+)")[0], errors="coerce")
    X["power_per_cc"] = X["power_bhp"] / X["engine_cc"]

    X["Volume"] = X["Length"] * X["Width"] * X["Height"]
    X["Model_clean"] = clean_model(X["Model"])
    X["Make"] = X["Make"].astype("string")
    X["Make_Fuel"] = base_model(X["Make"], X["Model_clean"])   # name kept for pipeline column lists

    for c in HIGH_CARD_COLS + LOW_CARD_COLS:
        X[c] = X[c].astype("object").where(X[c].notna(), np.nan)
    return X[NUM_COLS + HIGH_CARD_COLS + LOW_CARD_COLS]
