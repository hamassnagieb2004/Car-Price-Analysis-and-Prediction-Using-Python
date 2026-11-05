"""
================================================================================
CAR PRICE INTELLIGENCE | Streamlit Application
================================================================================
Purpose
-------
A multi-page SaaS-style application for estimating used-car prices, exploring
uploaded vehicle datasets, and reviewing public market-search results.

Architecture
------------
1. Presentation: Streamlit pages, responsive forms, theme-aware CSS and charts.
2. Inference: the user's serialized scikit-learn pipeline
   (car_price_best_pipeline.pkl), loaded with joblib.
3. Training parity: the custom preprocessing functions/transformers below
   reproduce the definitions used by the supplied training notebook. The
   saved pipeline therefore receives RAW feature rows, not manually encoded
   or scaled values.
4. Analytics: optional CSV upload for dataset-level EDA.
5. Market research: optional DuckDuckGo search integration (ddgs package).
6. Reliability: cached model loading, input validation, safe fallbacks and
   explicit errors for missing/incompatible model artifacts.

Run
---
pip install streamlit pandas numpy scikit-learn joblib plotly ddgs
streamlit run app.py

Place car_price_best_pipeline.pkl beside this file, or set MODEL_PATH in .env
or the environment. The pickle was created with custom objects in __main__;
their definitions are intentionally included below and MUST remain importable
before joblib.load() is called.

Security note
-------------
Only load trusted pickle/joblib files. Deserialization can execute code.
================================================================================
"""

from __future__ import annotations

import os
import re
import sys
import html
import logging
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    StandardScaler,
    TargetEncoder,
)
from sklearn.pipeline import Pipeline

# Optional market-search dependency. The app remains usable without it.
try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except Exception:
    try:
        from duckduckgo_search import DDGS  # legacy package name
        DDGS_AVAILABLE = True
    except Exception:
        DDGS = None
        DDGS_AVAILABLE = False


# =============================================================================
# Streamlit configuration and logging
# =============================================================================
st.set_page_config(
    page_title="AutoValue | Car Price Intelligence",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("autovalue")
CURRENT_YEAR = 2026
MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).with_name("car_price_best_pipeline.pkl")))


# =============================================================================
# TRAINING-NOTEBOOK PREPROCESSING DEFINITIONS
# Keep names/signatures compatible with the objects stored in the supplied PKL.
# =============================================================================
def cap_kilometer(X, threshold=500_000):
    """Cap Kilometer at threshold while preserving row count."""
    X = X.copy()
    if "Kilometer" in X.columns:
        X["Kilometer"] = pd.to_numeric(X["Kilometer"], errors="coerce").clip(upper=threshold)
    return X


def feature_engineering(X):
    X = X.copy()
    X["car_age"] = CURRENT_YEAR - pd.to_numeric(X["Year"], errors="coerce")
    X["km_per_year"] = pd.to_numeric(X["Kilometer"], errors="coerce") / X["car_age"].clip(lower=1)

    engine_text = X["Engine"].astype("string")
    engine_cc = engine_text.str.extract(r"(\d+(?:\.\d+)?)\s*cc\b", flags=re.IGNORECASE)[0]
    X["engine_cc"] = pd.to_numeric(engine_cc, errors="coerce")

    power_text = X["Max Power"].astype("string")
    power = power_text.str.extract(
        r"^\s*([\d.]+)\s*(?:bhp|BHP|PS|HP)?\s*@\s*([\d,]+)?", flags=re.IGNORECASE
    )
    X["power_bhp"] = pd.to_numeric(power[0], errors="coerce")
    X["power_rpm"] = pd.to_numeric(power[1].str.replace(",", "", regex=False), errors="coerce")

    torque_text = X["Max Torque"].astype("string")
    torque = torque_text.str.extract(
        r"^\s*([\d.]+)\s*(?:Nm|N\.m|kgm|kg-m)?\s*@\s*([\d,]+)?", flags=re.IGNORECASE
    )
    X["torque_nm"] = pd.to_numeric(torque[0], errors="coerce")
    X["torque_rpm"] = pd.to_numeric(torque[1].str.replace(",", "", regex=False), errors="coerce")

    X["Volume"] = (
        pd.to_numeric(X["Length"], errors="coerce")
        * pd.to_numeric(X["Width"], errors="coerce")
        * pd.to_numeric(X["Height"], errors="coerce")
    )
    return X


def impute_from_model(X, report=False):
    X = X.copy()
    X["_original_missing_count"] = X.isna().sum(axis=1)
    model = X["Model"].astype("string").fillna("")
    target_columns = [c for c in [
        "engine_cc", "power_bhp", "torque_nm", "Drivetrain", "Seating Capacity", "Fuel Type"
    ] if c in X.columns]
    missing_before = X[target_columns].isna().sum()

    for col in ["engine_cc", "power_bhp", "torque_nm", "Seating Capacity"]:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").astype("float64")

    if "engine_cc" in X.columns:
        match = model.str.extract(r"\b(\d+(?:\.\d+)?)\s*(L|cc)\b|\b(\d+\.\d+)\b", flags=re.IGNORECASE)
        value = pd.to_numeric(match[0].fillna(match[2]), errors="coerce")
        unit = match[1].str.lower()
        candidate = value.where(unit.eq("cc").fillna(False), value * 1000)
        candidate = candidate.mask(match[0].isna() & match[2].notna())
        mask = X["engine_cc"].isna() & candidate.notna() & candidate.between(500, 10000)
        X.loc[mask, "engine_cc"] = candidate.loc[mask]

    if "power_bhp" in X.columns:
        match = model.str.extract(r"\b(\d+(?:\.\d+)?)\s*(bhp|ps|hp)\b", flags=re.IGNORECASE)
        value, unit = pd.to_numeric(match[0], errors="coerce"), match[1].str.lower()
        candidate = value.where(unit.ne("ps").fillna(True), value * 0.98632)
        mask = X["power_bhp"].isna() & candidate.notna() & candidate.between(20, 2000)
        X.loc[mask, "power_bhp"] = candidate.loc[mask]

    if "torque_nm" in X.columns:
        match = model.str.extract(r"\b(\d+(?:\.\d+)?)\s*(Nm|N\.m)\b", flags=re.IGNORECASE)
        candidate = pd.to_numeric(match[0], errors="coerce")
        mask = X["torque_nm"].isna() & candidate.notna() & candidate.between(20, 5000)
        X.loc[mask, "torque_nm"] = candidate.loc[mask]

    if "Drivetrain" in X.columns:
        patterns = [
            (r"\bAWD\b", "AWD"), (r"\bFWD\b", "FWD"), (r"\bRWD\b", "RWD"),
            (r"\bquattro\b", "AWD"), (r"\bxDrive\b", "AWD"),
            (r"\b4MATIC\b", "AWD"), (r"\b4x4\b", "AWD"),
        ]
        candidate = pd.Series(pd.NA, index=X.index, dtype="string")
        for pattern, value in patterns:
            mask = model.str.contains(pattern, case=False, regex=True, na=False) & candidate.isna()
            candidate.loc[mask] = value
        mask = X["Drivetrain"].isna() & candidate.notna()
        X.loc[mask, "Drivetrain"] = candidate.loc[mask]

    if "Seating Capacity" in X.columns:
        match = model.str.extract(r"\b([2-9])\s*(?:STR|Seaters?|Seats)\b", flags=re.IGNORECASE)
        candidate = pd.to_numeric(match[0], errors="coerce")
        mask = X["Seating Capacity"].isna() & candidate.notna()
        X.loc[mask, "Seating Capacity"] = candidate.loc[mask]

    if "Fuel Type" in X.columns:
        candidate = pd.Series(pd.NA, index=X.index, dtype="string")
        for pattern, value in [(r"\bDiesel\b", "Diesel"), (r"\bPetrol\b", "Petrol")]:
            mask = model.str.contains(pattern, case=False, regex=True, na=False) & candidate.isna()
            candidate.loc[mask] = value
        mask = X["Fuel Type"].isna() & candidate.notna()
        X.loc[mask, "Fuel Type"] = candidate.loc[mask]

    if report:
        st.dataframe(pd.DataFrame({
            "Missing before": missing_before,
            "Missing after": X[target_columns].isna().sum(),
        }))
    return X


def drop_original_columns(X):
    return X.drop(
        columns=[c for c in ["Engine", "Max Power", "Max Torque", "Year", "_original_missing_count"] if c in X.columns],
        errors="ignore",
    ).copy()


def drop_rpm_features(X):
    return X.drop(columns=[c for c in ["power_rpm", "torque_rpm"] if c in X.columns]).copy()


def clean_model(X, report=True):
    X = X.copy()
    model = X["Model"].astype("string").str.lower()
    model = model.str.replace(r"\s*\[\s*\d{4}(?:\s*-\s*\d{4})?\s*\]", "", regex=True)
    model = model.str.replace(r"[‐-‒–—]", "-", regex=True)
    model = model.str.replace(r"\s*\+\s*", " + ", regex=True)
    model = model.str.replace(r"\s+", " ", regex=True).str.strip()
    model = model.str.replace(r"\b(xdrive|sdrive)\s+(\d)", r"\1\2", regex=True)
    X["Model_clean"] = model.str.replace(r"\s+", " ", regex=True).str.strip()
    return X.drop(columns=["Model"], errors="ignore")


REDUNDANT_COLS = [
    "Length", "Width", "Height", "torque_nm", "engine_cc",
    "Kilometer", "Seating Capacity",
]


def drop_redundant_columns(X):
    return X.drop(columns=[c for c in REDUNDANT_COLS if c in X.columns]).copy()


class ModelMakeImputer(BaseEstimator, TransformerMixin):
    def __init__(self, min_group_size=3, report=True):
        self.min_group_size, self.report = min_group_size, report
        self.numeric_cols = [
            "engine_cc", "power_bhp", "power_rpm", "torque_nm", "torque_rpm",
            "Length", "Width", "Height", "Seating Capacity", "Volume", "Fuel Tank Capacity",
        ]
        self.categorical_cols = ["Drivetrain"]

    @staticmethod
    def get_statistic(series, numeric=True):
        values = series.dropna()
        if values.empty:
            return np.nan
        if numeric:
            return values.median()
        modes = values.mode()
        return modes.iloc[0] if not modes.empty else np.nan

    def fit(self, X, y=None):
        X = X.copy()
        self.numeric_cols_ = [c for c in self.numeric_cols if c in X.columns]
        self.categorical_cols_ = [c for c in self.categorical_cols if c in X.columns]
        for col in self.numeric_cols_:
            X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
        self.group_stats_ = {}
        if {"Make", "Model_clean"}.issubset(X.columns):
            for col in self.numeric_cols_:
                self.group_stats_[col] = X.groupby(["Make", "Model_clean"], dropna=False)[col].agg(["median", "count"])
            for col in self.categorical_cols_:
                self.group_stats_[col] = X.groupby(["Make", "Model_clean"], dropna=False)[col].agg(
                    mode=lambda s: self.get_statistic(s, numeric=False), count="count"
                )
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.numeric_cols_:
            X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
        for col in self.numeric_cols_ + self.categorical_cols_:
            if col not in X.columns:
                continue
            if col in self.group_stats_ and {"Make", "Model_clean"}.issubset(X.columns):
                stats = self.group_stats_[col]
                for idx in X.index[X[col].isna()]:
                    key = (X.at[idx, "Make"], X.at[idx, "Model_clean"])
                    if key not in stats.index:
                        continue
                    row = stats.loc[key]
                    if row["count"] < self.min_group_size:
                        continue
                    value = row["median" if col in self.numeric_cols_ else "mode"]
                    if pd.notna(value):
                        X.at[idx, col] = value
        return X


class GroupBasedImputer(BaseEstimator, TransformerMixin):
    def __init__(self, min_group_size=3, report=True):
        self.min_group_size, self.report = min_group_size, report
        self.numeric_cols = [
            "engine_cc", "power_bhp", "power_rpm", "torque_nm", "torque_rpm",
            "Length", "Width", "Height", "Seating Capacity", "Volume", "Fuel Tank Capacity",
        ]
        self.categorical_cols = ["Drivetrain"]

    @staticmethod
    def get_statistic(series, numeric=True):
        values = series.dropna()
        if values.empty:
            return np.nan
        if numeric:
            return values.median()
        modes = values.mode()
        return modes.iloc[0] if not modes.empty else np.nan

    def fit(self, X, y=None):
        X = X.copy()
        self.numeric_cols_ = [c for c in self.numeric_cols if c in X.columns]
        self.categorical_cols_ = [c for c in self.categorical_cols if c in X.columns]
        for col in self.numeric_cols_:
            X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
        self.global_values_, self.make_values_, self.model_values_ = {}, {}, {}
        for col in self.numeric_cols_:
            self.global_values_[col] = self.get_statistic(X[col], True)
        for col in self.categorical_cols_:
            self.global_values_[col] = self.get_statistic(X[col], False)
        if "Make" in X.columns:
            for col in self.numeric_cols_:
                self.make_values_[col] = X.groupby("Make")[col].median()
            for col in self.categorical_cols_:
                self.make_values_[col] = X.groupby("Make")[col].agg(lambda s: self.get_statistic(s, False))
        if {"Make", "Model_clean"}.issubset(X.columns):
            for col in self.numeric_cols_:
                self.model_values_[col] = X.groupby(["Make", "Model_clean"])[col].agg(["median", "count"])
            for col in self.categorical_cols_:
                self.model_values_[col] = X.groupby(["Make", "Model_clean"])[col].agg(
                    mode=lambda s: self.get_statistic(s, False), count="count"
                )
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.numeric_cols_:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
        for col in self.numeric_cols_ + self.categorical_cols_:
            if col not in X.columns:
                continue
            numeric = col in self.numeric_cols_
            if col in self.model_values_ and {"Make", "Model_clean"}.issubset(X.columns):
                stats = self.model_values_[col]
                for idx in X.index[X[col].isna()]:
                    key = (X.at[idx, "Make"], X.at[idx, "Model_clean"])
                    if key in stats.index:
                        row = stats.loc[key]
                        if row["count"] >= self.min_group_size:
                            value = row["median" if numeric else "mode"]
                            if pd.notna(value):
                                X.at[idx, col] = value
            if col in self.make_values_ and "Make" in X.columns:
                stats = self.make_values_[col]
                for idx in X.index[X[col].isna()]:
                    value = stats.get(X.at[idx, "Make"], np.nan)
                    if pd.notna(value):
                        X.at[idx, col] = value
            fallback = self.global_values_.get(col, np.nan)
            if pd.notna(fallback):
                X[col] = X[col].fillna(fallback)
        return X


class YeoJohnsonTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.numeric_columns_ = [c for c in X.select_dtypes(include=["number"]).columns if c != "Price"]
        self.transformer_ = PowerTransformer(method="yeo-johnson", standardize=False)
        if self.numeric_columns_:
            self.transformer_.fit(X[self.numeric_columns_])
        return self

    def transform(self, X):
        X = X.copy()
        if self.numeric_columns_:
            X[self.numeric_columns_] = self.transformer_.transform(X[self.numeric_columns_])
        return X


class NumericScaler(BaseEstimator, TransformerMixin):
    def __init__(self, exclude=("Price",), scaler=None):
        self.exclude, self.scaler = exclude, scaler

    def fit(self, X, y=None):
        cols = X.select_dtypes(include="number").columns.tolist()
        self.columns_ = [c for c in cols if c not in self.exclude]
        base = self.scaler if self.scaler is not None else StandardScaler()
        self.scaler_ = clone(base)
        if self.columns_:
            self.scaler_.fit(X[self.columns_])
        return self

    def transform(self, X):
        X = X.copy()
        if getattr(self, "columns_", None):
            X[self.columns_] = self.scaler_.transform(X[self.columns_])
        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols, drop_cols=None, encoder=None):
        self.cat_cols, self.drop_cols, self.encoder = cat_cols, drop_cols, encoder

    def fit(self, X, y=None):
        X = X.copy()
        for col in self.drop_cols or []:
            X = X.drop(columns=col, errors="ignore")
        self.cat_cols_ = [c for c in self.cat_cols if c in X.columns]
        base = self.encoder if self.encoder is not None else OneHotEncoder(
            drop="first", sparse_output=False, handle_unknown="ignore"
        )
        self.encoder_ = clone(base)
        if self.cat_cols_:
            self.encoder_.fit(X[self.cat_cols_])
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.drop_cols or []:
            X = X.drop(columns=col, errors="ignore")
        if getattr(self, "cat_cols_", None):
            enc = self.encoder_.transform(X[self.cat_cols_])
            enc_df = pd.DataFrame(
                enc, columns=self.encoder_.get_feature_names_out(self.cat_cols_), index=X.index
            )
            X = X.drop(columns=self.cat_cols_)
            X = pd.concat([X, enc_df], axis=1)
        return X


def add_extra_features(X):
    X = X.copy()
    if "Kilometer" in X.columns:
        X["log_km"] = np.log1p(X["Kilometer"])
    X["power_per_cc"] = X["power_bhp"] / X["engine_cc"].replace(0, np.nan)
    first = X["Model_clean"].astype("string").str.split().str[0].fillna("")
    X["Make_Base"] = (X["Make"].astype("string").str.lower() + " " + first).str.strip()
    return X


# The training notebook serialized custom objects under the module name
# ``__main__``. Register this app's compatible definitions there explicitly;
# this also protects Streamlit execution contexts that use a script wrapper.
for _pickle_name in (
    "cap_kilometer", "feature_engineering", "impute_from_model",
    "drop_original_columns", "drop_rpm_features", "clean_model",
    "drop_redundant_columns", "ModelMakeImputer", "GroupBasedImputer",
    "YeoJohnsonTransformer", "NumericScaler", "CategoricalEncoder",
    "add_extra_features",
):
    setattr(sys.modules["__main__"], _pickle_name, globals()[_pickle_name])


# =============================================================================
# Theme system
# =============================================================================
THEMES = {
    "Light": {
        "bg": "#f4f7fb", "surface": "#ffffff", "text": "#172033",
        "muted": "#68758a", "accent": "#2563eb", "border": "#e3eaf3",
        "sidebar": "#ffffff", "hero": "linear-gradient(120deg,#eaf2ff,#f8fbff)",
    },
    "Dark": {
        "bg": "#0b1220", "surface": "#121d2e", "text": "#e8eef8",
        "muted": "#9eacc2", "accent": "#60a5fa", "border": "#26364d",
        "sidebar": "#101a2a", "hero": "linear-gradient(120deg,#14243d,#172c49)",
    },
    "Vibrant": {
        "bg": "#101323", "surface": "#191d35", "text": "#f5f3ff",
        "muted": "#b8b5d4", "accent": "#a78bfa", "border": "#343455",
        "sidebar": "#171a30", "hero": "linear-gradient(120deg,#34205f,#173e5e)",
    },
}


def apply_theme(theme_name: str) -> None:
    t = THEMES[theme_name]
    st.markdown(
        f"""
        <style>
        :root {{
          --app-bg:{t['bg']}; --surface:{t['surface']}; --text:{t['text']};
          --muted:{t['muted']}; --accent:{t['accent']}; --border:{t['border']};
        }}
        .stApp {{ background:var(--app-bg); color:var(--text); }}
        [data-testid="stSidebar"] {{ background:{t['sidebar']}; border-right:1px solid var(--border); }}
        [data-testid="stHeader"] {{ background:transparent; }}
        h1,h2,h3,h4,p,label,span {{ color:var(--text); }}
        .block-container {{ padding-top:2rem; padding-bottom:3rem; max-width:1500px; }}
        .hero {{
          background:{t['hero']}; border:1px solid var(--border); border-radius:22px;
          padding:28px 32px; margin-bottom:22px;
        }}
        .hero h1 {{ margin:0 0 8px 0; font-size:2.15rem; letter-spacing:-.04em; }}
        .hero p {{ color:var(--muted); margin:0; font-size:1rem; }}
        .eyebrow {{ color:var(--accent); font-size:.76rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; }}
        .panel {{
          background:var(--surface); border:1px solid var(--border); border-radius:18px;
          padding:18px 20px; margin-bottom:14px;
        }}
        .small-muted {{ color:var(--muted); font-size:.88rem; }}
        .price-value {{ font-size:2.4rem; font-weight:850; color:var(--accent); letter-spacing:-.04em; }}
        div[data-testid="stMetric"] {{
          background:var(--surface); border:1px solid var(--border);
          border-radius:16px; padding:14px 16px;
        }}
        div[data-testid="stMetricLabel"] p {{ color:var(--muted) !important; }}
        div.stButton > button, div.stDownloadButton > button {{
          border-radius:11px; border:1px solid var(--accent); font-weight:700;
          min-height:2.7rem;
        }}
        div.stButton > button[kind="primary"] {{
          background:var(--accent); color:white; border:0;
        }}
        input, textarea {{ border-radius:9px !important; }}
        .stTabs [data-baseweb="tab-list"] {{ gap:8px; }}
        .stTabs [data-baseweb="tab"] {{ border-radius:10px 10px 0 0; }}
        footer {{ visibility:hidden; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Model loading and prediction
# =============================================================================
@st.cache_resource(show_spinner="Loading the trained car valuation pipeline…")
def load_model(model_path: str):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}. Put car_price_best_pipeline.pkl beside app.py "
            "or set MODEL_PATH to its location."
        )
    # Custom classes/functions above must exist in __main__ before this call.
    return joblib.load(path)


def build_raw_vehicle(
    make: str, model: str, year: int, kilometer: float, fuel_type: str,
    transmission: str, engine_cc: float, max_power_bhp: float,
    drivetrain: str, location: str, color: str, owner: str, seller_type: str,
    torque_nm: float | None, length: float | None, width: float | None,
    height: float | None, seating: float | None, fuel_tank: float | None,
) -> pd.DataFrame:
    """Create a one-row RAW record with the exact column names used in training."""
    engine = f"{engine_cc:g} cc" if engine_cc and engine_cc > 0 else np.nan
    power = f"{max_power_bhp:g} bhp @ 5000 rpm" if max_power_bhp and max_power_bhp > 0 else np.nan
    torque = f"{torque_nm:g} Nm @ 2500 rpm" if torque_nm and torque_nm > 0 else np.nan
    row = {
        "Make": make, "Model": model, "Year": year, "Kilometer": kilometer,
        "Fuel Type": fuel_type, "Transmission": transmission,
        "Engine": engine, "Max Power": power, "Max Torque": torque,
        "Drivetrain": drivetrain, "Location": location, "Color": color,
        "Owner": owner, "Seller Type": seller_type,
        "Length": length, "Width": width, "Height": height,
        "Seating Capacity": seating, "Fuel Tank Capacity": fuel_tank,
    }
    return pd.DataFrame([row])


def predict_price(pipeline, raw_df: pd.DataFrame) -> float:
    pred = np.asarray(pipeline.predict(raw_df), dtype=float).reshape(-1)
    if len(pred) != 1 or not np.isfinite(pred[0]):
        raise ValueError("The model returned an invalid prediction.")
    # Notebook's TransformedTargetRegressor already applies expm1 internally.
    # Do NOT apply expm1 again here.
    return max(0.0, float(pred[0]))


# =============================================================================
# Reusable layout helpers
# =============================================================================
def page_header(eyebrow: str, title: str, subtitle: str):
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{html.escape(eyebrow)}</div>'
        f'<h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def money(value: float, currency: str = "INR") -> str:
    # Training notebook is based on an Indian used-car dataset; retain INR by default.
    if currency == "EGP":
        return f"EGP {value:,.0f}"
    if currency == "USD":
        return f"${value:,.0f}"
    return f"₹ {value:,.0f}"


def market_search(query: str, max_results: int = 8) -> list[dict[str, Any]]:
    if not DDGS_AVAILABLE:
        raise RuntimeError("Install the optional search package with: pip install ddgs")
    with DDGS() as client:
        results = list(client.text(query, max_results=max_results))
    return results


# =============================================================================
# Sidebar navigation
# =============================================================================
with st.sidebar:
    st.markdown("## 🚘 AutoValue")
    st.caption("CAR PRICE INTELLIGENCE")
    page = st.radio(
        "Workspace",
        ["Prediction Hub", "Data Insights & Analytics", "Web Search & Market Trends", "About / Help"],
        index=0,
        label_visibility="collapsed",
    )
    st.divider()
    theme_name = st.selectbox("Appearance", list(THEMES.keys()), index=0)
    currency = st.selectbox("Display currency", ["INR", "EGP", "USD"], index=0)
    st.divider()
    st.markdown("**Model status**")
    try:
        model_pipeline = load_model(str(MODEL_PATH))
        st.success("Pipeline loaded")
    except Exception as exc:
        model_pipeline = None
        st.error("Model unavailable")
        with st.expander("Technical details"):
            st.code(str(exc))
    st.caption("Model artifact: " + MODEL_PATH.name)
    st.caption("Valuation is an estimate, not a dealer quote.")

apply_theme(theme_name)


# =============================================================================
# PAGE 1 — PREDICTION HUB
# =============================================================================
if page == "Prediction Hub":
    page_header(
        "VALUATION WORKSPACE",
        "Know what your car is worth.",
        "Enter the vehicle specification to generate an estimate from the trained machine-learning pipeline.",
    )

    if model_pipeline is None:
        st.warning(
            "Prediction is disabled until the trained pipeline is available. "
            "Place `car_price_best_pipeline.pkl` next to `app.py` and restart the app."
        )

    left, right = st.columns([1.45, 0.85], gap="large")
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Vehicle profile")
        with st.form("prediction_form", clear_on_submit=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                make = st.text_input("Make", value="Maruti Suzuki", help="Manufacturer, e.g. Toyota, Honda, BMW.")
                model_name = st.text_input("Model / Variant", value="Swift VXI")
                year = st.number_input("Manufacturing year", min_value=1990, max_value=CURRENT_YEAR, value=2018, step=1)
                kilometer = st.number_input("Kilometers driven", min_value=0, max_value=1_000_000, value=65000, step=1000)
            with c2:
                fuel_type = st.selectbox("Fuel type", ["Petrol", "Diesel", "CNG", "LPG", "Electric", "Hybrid", "Other"])
                transmission = st.selectbox("Transmission", ["Manual", "Automatic", "AMT", "CVT", "DCT", "Other"])
                engine_cc = st.number_input("Engine capacity (cc)", min_value=0, max_value=10000, value=1197, step=50)
                max_power = st.number_input("Maximum power (bhp)", min_value=0.0, max_value=2000.0, value=82.0, step=1.0)
            with c3:
                drivetrain = st.selectbox("Drivetrain", ["FWD", "RWD", "AWD", "4WD", "Other"])
                location = st.text_input("Location / City", value="Delhi")
                color = st.selectbox("Color", ["White", "Silver", "Grey", "Black", "Red", "Blue", "Brown", "Other"])
                owner = st.selectbox("Number of owners", ["First", "Second", "Third", "Fourth", "4 or More", "UnRegistered Car"])
            seller_type = st.radio("Seller type", ["Individual", "Dealer", "Trustmark Dealer"], horizontal=True)
            with st.expander("Advanced specifications (optional but helpful)"):
                a1, a2, a3 = st.columns(3)
                with a1:
                    torque_nm = st.number_input("Maximum torque (Nm)", min_value=0.0, max_value=5000.0, value=0.0, step=5.0)
                    length = st.number_input("Length (mm)", min_value=0.0, max_value=10000.0, value=0.0, step=10.0)
                with a2:
                    width = st.number_input("Width (mm)", min_value=0.0, max_value=5000.0, value=0.0, step=10.0)
                    height = st.number_input("Height (mm)", min_value=0.0, max_value=5000.0, value=0.0, step=10.0)
                with a3:
                    seating = st.number_input("Seating capacity", min_value=0, max_value=20, value=0, step=1)
                    fuel_tank = st.number_input("Fuel tank capacity (L)", min_value=0.0, max_value=500.0, value=0.0, step=1.0)
            submitted = st.form_submit_button("✨ Estimate car value", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("How the estimate works")
        st.markdown(
            "- **Raw inputs** are passed to the saved pipeline.\n"
            "- Feature engineering extracts engine, power, age and usage signals.\n"
            "- Group-based imputation handles missing specifications.\n"
            "- Categorical encoding and model inference run in the fitted pipeline.\n"
            "- The target transform is reversed inside `TransformedTargetRegressor`."
        )
        st.info("The training notebook uses an Indian used-car dataset and reports prices in INR. Select another display currency only if you have applied an appropriate exchange rate.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Input quality checklist")
        st.markdown(
            "✓ Correct make and model spelling  \n"
            "✓ Realistic odometer reading  \n"
            "✓ Correct year and engine size  \n"
            "✓ Optional dimensions / torque if available"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        if model_pipeline is None:
            st.error("Model is not loaded; prediction cannot run.")
        elif not make.strip() or not model_name.strip():
            st.error("Make and Model are required.")
        elif year > CURRENT_YEAR or kilometer < 0:
            st.error("Please review the year and kilometer values.")
        else:
            raw = build_raw_vehicle(
                make.strip(), model_name.strip(), int(year), float(kilometer),
                fuel_type, transmission, float(engine_cc), float(max_power),
                drivetrain, location.strip() or "Unknown", color, owner, seller_type,
                float(torque_nm) if torque_nm > 0 else None,
                float(length) if length > 0 else None,
                float(width) if width > 0 else None,
                float(height) if height > 0 else None,
                float(seating) if seating > 0 else None,
                float(fuel_tank) if fuel_tank > 0 else None,
            )
            try:
                with st.spinner("Analyzing vehicle specification…"):
                    predicted = predict_price(model_pipeline, raw)
                st.session_state["last_prediction"] = {
                    "price": predicted, "raw": raw, "make": make, "model": model_name,
                    "year": int(year), "kilometer": float(kilometer),
                }
                st.success("Valuation generated successfully.")
                p1, p2, p3 = st.columns(3)
                p1.metric("Estimated value", money(predicted, currency))
                p2.metric("Vehicle age", f"{CURRENT_YEAR - int(year)} years")
                p3.metric("Mileage", f"{kilometer:,.0f} km")
                st.markdown(
                    f'<div class="panel"><div class="small-muted">MODEL ESTIMATE</div>'
                    f'<div class="price-value">{money(predicted, currency)}</div>'
                    f'<div class="small-muted">{html.escape(make)} {html.escape(model_name)} · {year} · {kilometer:,.0f} km</div></div>',
                    unsafe_allow_html=True,
                )
                with st.expander("Inspect the raw feature row sent to the model"):
                    st.dataframe(raw, use_container_width=True)
                st.download_button(
                    "Download valuation input (CSV)",
                    data=raw.to_csv(index=False).encode("utf-8"),
                    file_name="vehicle_valuation_input.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                LOGGER.exception("Prediction failed")
                st.error("Prediction failed. The model artifact and preprocessing dependencies may be incompatible.")
                with st.expander("Technical details"):
                    st.code(f"{type(exc).__name__}: {exc}")


# =============================================================================
# PAGE 2 — DATA INSIGHTS & ANALYTICS
# =============================================================================
elif page == "Data Insights & Analytics":
    page_header(
        "DATA EXPLORER",
        "Understand the market behind the model.",
        "Upload the source car dataset to explore price distributions, brand mix, mileage, fuel types and relationships.",
    )
    uploaded = st.file_uploader("Upload a car dataset (CSV)", type=["csv"], help="Expected columns include Price, Make, Year, Kilometer, Fuel Type and Transmission.")
    if uploaded is None:
        st.info("Upload your `car details v4.csv` (or another compatible CSV) to populate the analytics dashboard. No synthetic data is presented as real market data.")
    else:
        try:
            data = pd.read_csv(uploaded)
            st.session_state["analytics_df"] = data
        except Exception as exc:
            st.error(f"Could not read this CSV: {exc}")
            data = None

    data = st.session_state.get("analytics_df")
    if data is not None:
        st.caption(f"Dataset loaded: {len(data):,} rows × {len(data.columns):,} columns")
        if "Price" not in data.columns:
            st.warning("The dataset needs a `Price` column for price analytics.")
        else:
            filtered = data.copy()
            fcols = st.columns(4)
            if "Make" in filtered.columns:
                makes = sorted(filtered["Make"].dropna().astype(str).unique())
                selected_makes = fcols[0].multiselect("Make filter", makes, default=[])
                if selected_makes:
                    filtered = filtered[filtered["Make"].astype(str).isin(selected_makes)]
            if "Fuel Type" in filtered.columns:
                fuels = sorted(filtered["Fuel Type"].dropna().astype(str).unique())
                selected_fuels = fcols[1].multiselect("Fuel filter", fuels, default=[])
                if selected_fuels:
                    filtered = filtered[filtered["Fuel Type"].astype(str).isin(selected_fuels)]
            if "Transmission" in filtered.columns:
                trans = sorted(filtered["Transmission"].dropna().astype(str).unique())
                selected_trans = fcols[2].multiselect("Transmission", trans, default=[])
                if selected_trans:
                    filtered = filtered[filtered["Transmission"].astype(str).isin(selected_trans)]
            price_values = pd.to_numeric(filtered["Price"], errors="coerce").dropna()
            if len(price_values):
                lo, hi = float(price_values.min()), float(price_values.max())
                selected_range = fcols[3].slider("Price range", min_value=lo, max_value=hi, value=(lo, hi))
                filtered = filtered[pd.to_numeric(filtered["Price"], errors="coerce").between(*selected_range)]

            prices = pd.to_numeric(filtered["Price"], errors="coerce").dropna()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Vehicles", f"{len(filtered):,}")
            m2.metric("Median price", f"{prices.median():,.0f}" if len(prices) else "—")
            m3.metric("Average price", f"{prices.mean():,.0f}" if len(prices) else "—")
            m4.metric("Price records", f"{prices.notna().sum():,}")

            tab1, tab2, tab3 = st.tabs(["Price distribution", "Market composition", "Relationships"])
            with tab1:
                if len(prices):
                    fig = px.histogram(filtered, x="Price", nbins=45, marginal="box", title="Vehicle price distribution")
                    st.plotly_chart(fig, use_container_width=True)
                    if "Make" in filtered.columns:
                        top = filtered.groupby("Make", dropna=True)["Price"].median().nlargest(12).reset_index()
                        fig2 = px.bar(top, x="Make", y="Price", title="Top makes by median listed price")
                        st.plotly_chart(fig2, use_container_width=True)
            with tab2:
                c1, c2 = st.columns(2)
                for col, container in [("Fuel Type", c1), ("Transmission", c2)]:
                    if col in filtered.columns:
                        counts = filtered[col].fillna("Unknown").astype(str).value_counts().head(12).rename_axis(col).reset_index(name="Vehicles")
                        container.plotly_chart(px.pie(counts, names=col, values="Vehicles", hole=.55, title=f"Mix by {col}"), use_container_width=True)
            with tab3:
                if {"Year", "Price"}.issubset(filtered.columns):
                    plot_data = filtered.copy()
                    plot_data["Price"] = pd.to_numeric(plot_data["Price"], errors="coerce")
                    plot_data["Year"] = pd.to_numeric(plot_data["Year"], errors="coerce")
                    fig = px.scatter(plot_data, x="Year", y="Price", color="Fuel Type" if "Fuel Type" in plot_data else None, hover_data=[c for c in ["Make", "Model"] if c in plot_data], title="Price vs manufacturing year", opacity=.65)
                    st.plotly_chart(fig, use_container_width=True)
                if {"Kilometer", "Price"}.issubset(filtered.columns):
                    plot_data = filtered.copy()
                    plot_data["Kilometer"] = pd.to_numeric(plot_data["Kilometer"], errors="coerce")
                    plot_data["Price"] = pd.to_numeric(plot_data["Price"], errors="coerce")
                    st.plotly_chart(px.scatter(plot_data, x="Kilometer", y="Price", title="Price vs kilometers driven", opacity=.55), use_container_width=True)
            with st.expander("Preview filtered data"):
                st.dataframe(filtered.head(500), use_container_width=True)
            st.download_button("Download filtered dataset", filtered.to_csv(index=False).encode("utf-8"), "filtered_car_data.csv", "text/csv")


# =============================================================================
# PAGE 3 — WEB SEARCH & MARKET TRENDS
# =============================================================================
elif page == "Web Search & Market Trends":
    page_header(
        "MARKET PULSE",
        "Explore current listings and market commentary.",
        "Search the public web for recent information related to a make, model, year or local used-car market.",
    )
    last = st.session_state.get("last_prediction", {})
    default_query = f"{last.get('make', 'Toyota')} {last.get('model', 'Corolla')} used car price"
    with st.form("market_search_form"):
        q1, q2 = st.columns([3, 1])
        query = q1.text_input("Search topic", value=default_query, help="Example: Toyota Corolla 2018 used price Delhi")
        region = q2.selectbox("Market", ["India", "Egypt", "United States", "United Kingdom", "Global"])
        limit = st.slider("Number of results", 3, 12, 7)
        search_clicked = st.form_submit_button("🔎 Search market", type="primary")
    if search_clicked:
        full_query = f"{query} used car prices {region}"
        if not DDGS_AVAILABLE:
            st.warning("Live search is not enabled in this environment. Install `ddgs` with `pip install ddgs` and restart the app.")
            st.markdown("**Suggested manual search:**")
            st.code(full_query)
        else:
            try:
                with st.spinner("Searching public web results…"):
                    results = market_search(full_query, limit)
                if not results:
                    st.info("No results were returned. Try a broader query.")
                else:
                    st.session_state["market_results"] = results
                    for idx, item in enumerate(results, start=1):
                        title = item.get("title") or "Search result"
                        href = item.get("href") or item.get("url") or ""
                        body = item.get("body") or item.get("snippet") or ""
                        st.markdown(
                            f'<div class="panel"><div class="eyebrow">RESULT {idx}</div>'
                            f'<h3>{html.escape(title)}</h3><p class="small-muted">{html.escape(body)}</p>'
                            f'<a href="{html.escape(href, quote=True)}" target="_blank" rel="noopener noreferrer">Open source ↗</a></div>',
                            unsafe_allow_html=True,
                        )
            except Exception as exc:
                st.error(f"Market search failed: {exc}")
                st.caption("Some networks block search providers. Try again later or use the manual query.")
    cached_results = st.session_state.get("market_results", [])
    if cached_results and not search_clicked:
        st.caption("Showing the most recent search results from this session.")
        for item in cached_results:
            title = item.get("title") or "Search result"
            href = item.get("href") or item.get("url") or ""
            st.markdown(f"- [{title}]({href})" if href else f"- {title}")

    st.warning("Web search results are third-party information, not verified transaction prices. Confirm vehicle condition, taxes, mileage and local comparables before making a purchase decision.")


# =============================================================================
# PAGE 4 — ABOUT / HELP
# =============================================================================
elif page == "About / Help":
    page_header(
        "MODEL DOCUMENTATION",
        "A transparent view of the valuation engine.",
        "How the saved pipeline transforms vehicle specifications into an estimated resale price.",
    )
    st.subheader("Model architecture")
    st.markdown(
        """
        The supplied artifact is a **scikit-learn Pipeline** trained in the attached
        notebook. Its selected model is **HistGradientBoostingRegressor** wrapped in
        **TransformedTargetRegressor**. Although the request mentions Linear Regression,
        the uploaded `car_price_best_pipeline.pkl` is the improved tree-based model,
        not the earlier Linear Regression baseline.
        """
    )
    st.info("The notebook compares Linear Regression, Random Forest and HistGradientBoosting. HistGradientBoosting is selected as the best model by test-set R²_log (0.965), with MedAPE 7.959% and within-15% accuracy 75.0% on that notebook's test split.")
    with st.expander("Preprocessing sequence"):
        st.markdown(
            """
            1. Cap extreme `Kilometer` values at 500,000.
            2. Engineer `car_age`, `km_per_year`, `engine_cc`, `power_bhp`,
               `power_rpm`, `torque_nm`, `torque_rpm` and vehicle `Volume`.
            3. Parse additional specification clues from the model name.
            4. Normalize the model string and apply Make+Model, Make-level and
               global imputation fallbacks.
            5. Add `log_km`, `power_per_cc` and `Make_Base`.
            6. Encode high-cardinality fields with `TargetEncoder`, low-cardinality
               fields with `OrdinalEncoder`, then predict.
            """
        )
    with st.expander("Target transformation and price units"):
        st.markdown(
            """
            The training pipeline uses `TransformedTargetRegressor` with
            `func=np.log1p` and `inverse_func=np.expm1`. Its `predict()` method
            already returns prices on the original target scale. The app therefore
            **does not call `expm1` a second time**.

            The training notebook uses an Indian used-car dataset. Its native
            price unit is INR (Indian Rupees). The currency selector changes only
            the display label; it does not perform foreign-exchange conversion.
            """
        )
    with st.expander("Important model limitations"):
        st.markdown(
            """
            - Predictions depend on the training dataset's market, time period and coverage.
            - Make/model spelling and specification quality can affect the estimate.
            - Missing optional specifications are imputed using learned training statistics.
            - A prediction is not a certified appraisal or a guaranteed selling price.
            - The notebook's test metrics are historical evaluation results, not a
              promise of performance on new data.
            """
        )
    st.subheader("Troubleshooting")
    st.markdown(
        """
        **Model file not found:** place `car_price_best_pipeline.pkl` in the same
        directory as `app.py`, or set `MODEL_PATH`.

        **Pickle / scikit-learn compatibility error:** install the same or a
        compatible scikit-learn version used when the artifact was created.
        Pickled estimators are not guaranteed to work across arbitrary library versions.

        **Missing custom transformer:** keep the notebook-derived functions and
        classes in this file before `joblib.load()`.

        **Search unavailable:** install `ddgs` or use the displayed manual search query.
        """
    )
    st.caption("AutoValue · Streamlit interface for the supplied car-price model artifact.")
