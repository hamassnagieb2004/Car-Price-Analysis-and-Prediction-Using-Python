"""Preprocessing + pipelines. Cleaning/imputation/transformer code is taken as-is from the original notebook."""
import re
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor, make_column_selector
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, PowerTransformer, StandardScaler, RobustScaler,
                                   OneHotEncoder, OrdinalEncoder, TargetEncoder)
try:
    from IPython.display import display
except ImportError:                       # running as a plain script
    display = print


# ============================================================
# Kilometer outliers
# ------------------------------------------------------------
# - drop_high_kilometer : DELETES rows (TRAIN ONLY, done outside the
#                         pipeline because a pipeline step must never
#                         change the number of rows / break X-y alignment).
# - cap_kilometer       : row-preserving cap, used INSIDE the pipeline so
#                         test / new data with absurd mileage still works.
# ============================================================
def drop_high_kilometer(X, y, threshold=500_000):
    """Delete rows where Kilometer > threshold from BOTH X and y."""
    if "Kilometer" not in X.columns:
        print("'Kilometer' not found - skipping.")
        return X, y
    mask = ~(X["Kilometer"] > threshold)          # NaN is kept
    print(f"Kilometer > {threshold:,}: dropped {int((~mask).sum())} rows")
    return X.loc[mask].copy(), y.loc[mask].copy()


def cap_kilometer(X, threshold=500_000):
    """Cap Kilometer at `threshold` (keeps all rows)."""
    X = X.copy()
    if "Kilometer" in X.columns:
        X["Kilometer"] = X["Kilometer"].clip(upper=threshold)
    return X

# ============================================================
# Drop rows that had >= 9 original missing fields
# (likely junk rows with almost no info)
# ============================================================
def prepare_rows(X, y, threshold=9):
    X = X.copy()
    X["_original_missing_count"] = X.isna().sum(axis=1)
    mask = X["_original_missing_count"] < threshold
    X_clean = X.loc[mask].drop(columns=["_original_missing_count"]).copy()
    y_clean = y.loc[X_clean.index].copy()
    print(f"Dropped: {int((~mask).sum())} | Kept: {int(mask.sum())}")
    return X_clean, y_clean

# ============================================================
# Custom feature engineering
# Creates:
#   - car_age       = current year - year
#   - km_per_year   = Kilometer / car_age
#   - engine_cc     = numeric cc extracted from "Engine" text
#   - power_bhp     = numeric bhp from "Max Power" text
#   - power_rpm     = rpm value from "Max Power" text
#   - torque_nm     = numeric Nm from "Max Torque" text
#   - torque_rpm    = rpm value from "Max Torque" text
#   - Volume        = Length × Width × Height
# ============================================================
def feature_engineering(X):
    X = X.copy()
    CURRENT_YEAR = 2026

    # 1. Car age
    X["car_age"] = CURRENT_YEAR - X["Year"]

    # 2. Kilometers per year (clip to avoid division by zero)
    X["km_per_year"] = X["Kilometer"] / X["car_age"].clip(lower=1)

    # 3. Engine capacity in cc
    engine_text = X["Engine"].astype("string")
    engine_cc = engine_text.str.extract(
        r"(\d+(?:\.\d+)?)\s*cc\b", flags=re.IGNORECASE
    )[0]
    X["engine_cc"] = pd.to_numeric(engine_cc, errors="coerce")

    # 4. Maximum power (bhp) and its RPM
    power_text = X["Max Power"].astype("string")
    power = power_text.str.extract(
        r"^\s*([\d.]+)\s*(?:bhp|BHP|PS|HP)?\s*@\s*([\d,]+)?",
        flags=re.IGNORECASE
    )
    X["power_bhp"] = pd.to_numeric(power[0], errors="coerce")
    X["power_rpm"] = pd.to_numeric(
        power[1].str.replace(",", "", regex=False), errors="coerce"
    )

    # 5. Maximum torque (Nm) and its RPM
    torque_text = X["Max Torque"].astype("string")
    torque = torque_text.str.extract(
        r"^\s*([\d.]+)\s*(?:Nm|N\.m|kgm|kg-m)?\s*@\s*([\d,]+)?",
        flags=re.IGNORECASE
    )
    X["torque_nm"] = pd.to_numeric(torque[0], errors="coerce")
    X["torque_rpm"] = pd.to_numeric(
        torque[1].str.replace(",", "", regex=False), errors="coerce"
    )

    # 6. Volume = Length × Width × Height
    X["Volume"] = X["Length"] * X["Width"] * X["Height"]

    return X

# ============================================================
# Impute missing values by parsing the "Model" string
# (e.g., "1.2L" → engine_cc=1200, "quattro" → AWD, "7 STR" → 7 seats)
# ============================================================
def impute_from_model(X, report=False):
    X = X.copy()
    X["_original_missing_count"] = X.isna().sum(axis=1)
    model = X["Model"].astype("string").fillna("")

    target_columns = [c for c in [
        "engine_cc", "power_bhp", "torque_nm",
        "Drivetrain", "Seating Capacity", "Fuel Type"
    ] if c in X.columns]

    missing_before = X[target_columns].isna().sum()

    # Force numeric columns to accept floats
    for col in ["engine_cc", "power_bhp", "torque_nm", "Seating Capacity"]:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").astype("float64").astype("float64")

    # 1. Engine capacity from Model (handles both "1.2 L" and "1200 cc")
    if "engine_cc" in X.columns:
        engine_match = model.str.extract(
            r"\b(\d+(?:\.\d+)?)\s*(L|cc)\b|\b(\d+\.\d+)\b",
            flags=re.IGNORECASE
        )
        engine_value  = pd.to_numeric(engine_match[0].fillna(engine_match[2]),
                                       errors="coerce").astype("float64")
        engine_unit   = engine_match[1].str.lower()
        engine_candidate = engine_value.where(
            engine_unit.eq("cc").fillna(False).astype(bool), engine_value * 1000
        )
        bare_decimal = engine_match[0].isna() & engine_match[2].notna()
        engine_candidate = engine_candidate.mask(bare_decimal)

        fill_mask = (
            X["engine_cc"].isna() & engine_candidate.notna()
            & engine_candidate.between(500, 10000)
        )
        X.loc[fill_mask, "engine_cc"] = engine_candidate.loc[fill_mask]

    # 2. Max power (bhp) — convert PS → BHP by ×0.98632
    if "power_bhp" in X.columns:
        power_match = model.str.extract(
            r"\b(\d+(?:\.\d+)?)\s*(bhp|ps|hp)\b", flags=re.IGNORECASE
        )
        power_value = pd.to_numeric(power_match[0], errors="coerce").astype("float64")
        power_unit  = power_match[1].str.lower()
        power_candidate = power_value.where(
            power_unit.ne("ps").fillna(True).astype(bool), power_value * 0.98632
        )
        fill_mask = (
            X["power_bhp"].isna() & power_candidate.notna()
            & power_candidate.between(20, 2000)
        )
        X.loc[fill_mask, "power_bhp"] = power_candidate.loc[fill_mask]

    # 3. Max torque (Nm)
    if "torque_nm" in X.columns:
        torque_match = model.str.extract(
            r"\b(\d+(?:\.\d+)?)\s*(Nm|N\.m)\b", flags=re.IGNORECASE
        )
        torque_candidate = pd.to_numeric(torque_match[0], errors="coerce").astype("float64")
        fill_mask = (
            X["torque_nm"].isna() & torque_candidate.notna()
            & torque_candidate.between(20, 5000)
        )
        X.loc[fill_mask, "torque_nm"] = torque_candidate.loc[fill_mask]

    # 4. Drivetrain from brand keywords
    if "Drivetrain" in X.columns:
        drivetrain_patterns = [
            (r"\bAWD\b", "AWD"), (r"\bFWD\b", "FWD"), (r"\bRWD\b", "RWD"),
            (r"\bquattro\b", "AWD"), (r"\bxDrive\b", "AWD"),
            (r"\b4MATIC\b", "AWD"), (r"\b4x4\b", "AWD"),
        ]
        drivetrain_candidate = pd.Series(pd.NA, index=X.index, dtype="string")
        for pattern, value in drivetrain_patterns:
            matches = model.str.contains(pattern, case=False, regex=True,
                                          na=False)
            drivetrain_candidate.loc[
                matches & drivetrain_candidate.isna()
            ] = value
        fill_mask = X["Drivetrain"].isna() & drivetrain_candidate.notna()
        X.loc[fill_mask, "Drivetrain"] = drivetrain_candidate.loc[fill_mask]

    # 5. Seating capacity from "7 STR" etc.
    if "Seating Capacity" in X.columns:
        seating_match = model.str.extract(
            r"\b([2-9])\s*(?:STR|Seaters?|Seats)\b", flags=re.IGNORECASE
        )
        seating_candidate = pd.to_numeric(seating_match[0], errors="coerce").astype("float64")
        fill_mask = (X["Seating Capacity"].isna()
                     & seating_candidate.notna())
        X.loc[fill_mask, "Seating Capacity"] = seating_candidate.loc[fill_mask]

    # 6. Fuel type from explicit names
    if "Fuel Type" in X.columns:
        fuel_patterns = [(r"\bDiesel\b", "Diesel"), (r"\bPetrol\b", "Petrol")]
        fuel_candidate = pd.Series(pd.NA, index=X.index, dtype="string")
        for pattern, value in fuel_patterns:
            matches = model.str.contains(pattern, case=False, regex=True,
                                          na=False)
            fuel_candidate.loc[matches & fuel_candidate.isna()] = value
        fill_mask = X["Fuel Type"].isna() & fuel_candidate.notna()
        X.loc[fill_mask, "Fuel Type"] = fuel_candidate.loc[fill_mask]

    # Missing report
    missing_after = X[target_columns].isna().sum()
    report_df = pd.DataFrame({
        "Missing_Before": missing_before,
        "Missing_After":  missing_after,
        "Values_Filled":  missing_before - missing_after
    })

    if report:
        print("\n" + "=" * 55)
        print("MISSING VALUES REPORT")
        print("=" * 55)
        display(report_df)

    return X

# ============================================================
# Drop raw text columns after extracting features
# (Year dropped too since we now have car_age)
# ============================================================
def drop_original_columns(X):
    X = X.copy()
    columns_to_drop = [
        "Engine", "Max Power", "Max Torque",
        "Year", "_original_missing_count"
    ]
    X = X.drop(
        columns=[c for c in columns_to_drop if c in X.columns],
        errors="ignore"
    )
    return X

# ============================================================
# Drop power_rpm and torque_rpm (highly correlated with
# their Nm/bhp counterparts, add noise more than signal)
# ============================================================
def drop_rpm_features(X):
    X = X.copy()
    columns_to_drop = ["power_rpm", "torque_rpm"]
    return X.drop(
        columns=[col for col in columns_to_drop if col in X.columns]
    )

# ============================================================
# Drop redundant / weak features (decided in section 8 from the
# correlation + VIF analysis). Kept as a pipeline step so the
# saved pipeline accepts RAW data.
#   - Keep Volume, drop Length / Width / Height
#   - Keep power_bhp, drop engine_cc / torque_nm
#   - Keep km_per_year, drop Kilometer
#   - Drop Seating Capacity (|r| < 0.05)
# ============================================================
REDUNDANT_COLS = [
    "Length", "Width", "Height",
    "torque_nm", "engine_cc",
    "Kilometer",
    "Seating Capacity",
]

def drop_redundant_columns(X):
    return X.drop(columns=[c for c in REDUNDANT_COLS if c in X.columns])

# ============================================================
# Normalize the "Model" string:
#   - remove year ranges like "[2017-2019]"
#   - unify dashes and spacing
#   - merge "xDrive 30d" → "xdrive30d"
# Result: 912 → 871 unique models (fewer = better encoding)
# ============================================================
def clean_model(X, report=True):
    X = X.copy()
    original = X["Model"].astype("string")
    model = original.str.lower()

    # 1. Remove year ranges [2016-2020]
    model = model.str.replace(
        r"\s*\[\s*\d{4}(?:\s*-\s*\d{4})?\s*\]", "", regex=True
    )
    # 2. Normalize dash characters
    model = model.str.replace(r"[‐-‒–—]", "-", regex=True)
    # 3. Normalize spaces around "+"
    model = model.str.replace(r"\s*\+\s*", " + ", regex=True)
    # 4. Collapse repeated whitespace
    model = model.str.replace(r"\s+", " ", regex=True).str.strip()
    # 5. Merge BMW drive designations
    model = model.str.replace(r"\b(xdrive|sdrive)\s+(\d)", r"\1\2", regex=True)
    # 6. Final cleanup
    model = model.str.replace(r"\s+", " ", regex=True).str.strip()

    X["Model_clean"] = model
    X = X.drop(columns=["Model"])

    if report:
        print("=" * 65)
        print("MODEL CLEANING REPORT")
        print("=" * 65)
        print(f"Unique models before: {original.nunique(dropna=True)}")
        print(f"Unique models after:  {model.nunique(dropna=True)}")
        print("Reduction in unique categories:",
              original.nunique(dropna=True) - model.nunique(dropna=True))
    return X

# ============================================================
# Custom Transformer — impute missing values using
# median (numeric) or mode (categorical) grouped by Make+Model.
# Only uses groups with >= min_group_size samples (avoids noise).
# ============================================================
class ModelMakeImputer(BaseEstimator, TransformerMixin):

    def __init__(self, min_group_size=3, report=True):
        self.min_group_size = min_group_size
        self.report = report
        self.numeric_cols = [
            "engine_cc", "power_bhp", "power_rpm",
            "torque_nm", "torque_rpm",
            "Length", "Width", "Height", "Seating Capacity",
            "Volume", "Fuel Tank Capacity"
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
                self.group_stats_[col] = X.groupby(
                    ["Make", "Model_clean"], dropna=False
                )[col].agg(["median", "count"])
            for col in self.categorical_cols_:
                self.group_stats_[col] = X.groupby(
                    ["Make", "Model_clean"], dropna=False
                )[col].agg(
                    mode=lambda s: self.get_statistic(s, numeric=False),
                    count="count"
                )
        return self

    def transform(self, X):
        X = X.copy()
        report_rows = []

        for col in self.numeric_cols_:
            X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)

        for col in self.numeric_cols_ + self.categorical_cols_:
            missing_before = int(X[col].isna().sum())
            filled = 0

            if col in self.group_stats_ and \
               {"Make", "Model_clean"}.issubset(X.columns):
                stats = self.group_stats_[col]
                for idx in X.index[X[col].isna()]:
                    key = (X.at[idx, "Make"], X.at[idx, "Model_clean"])
                    if key not in stats.index:
                        continue
                    row = stats.loc[key]
                    if row["count"] < self.min_group_size:
                        continue
                    stat_name = ("median" if col in self.numeric_cols_
                                 else "mode")
                    value = row[stat_name]
                    if pd.notna(value):
                        X.at[idx, col] = value
                        filled += 1

            report_rows.append({
                "Feature": col, "Missing_Before": missing_before,
                "Filled_From_Model_Make": filled,
                "Missing_After": int(X[col].isna().sum())
            })

        self.report_ = pd.DataFrame(report_rows)
        if self.report:
            print("\nMODEL + MAKE IMPUTATION REPORT")
            display(self.report_)
        return X

# ============================================================
# Fallback imputer — tries Model→Make→Global for
# any values still missing after Cell 25.
# ============================================================
class GroupBasedImputer(BaseEstimator, TransformerMixin):

    def __init__(self, min_group_size=3, report=True):
        self.min_group_size = min_group_size
        self.report = report
        self.numeric_cols = [
            "engine_cc", "power_bhp", "power_rpm",
            "torque_nm", "torque_rpm",
            "Length", "Width", "Height", "Seating Capacity",
            "Volume", "Fuel Tank Capacity"
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

        # Global fallback
        for col in self.numeric_cols_:
            self.global_values_[col] = self.get_statistic(X[col], True)
        for col in self.categorical_cols_:
            self.global_values_[col] = self.get_statistic(X[col], False)

        # Make-level stats
        if "Make" in X.columns:
            for col in self.numeric_cols_:
                self.make_values_[col] = X.groupby("Make")[col].median()
            for col in self.categorical_cols_:
                self.make_values_[col] = X.groupby("Make")[col].agg(
                    lambda s: self.get_statistic(s, False)
                )

        # Make+Model stats
        if {"Make", "Model_clean"}.issubset(X.columns):
            groups = ["Make", "Model_clean"]
            for col in self.numeric_cols_:
                self.model_values_[col] = X.groupby(groups)[col].agg(
                    ["median", "count"]
                )
            for col in self.categorical_cols_:
                self.model_values_[col] = X.groupby(groups)[col].agg(
                    mode=lambda s: self.get_statistic(s, False),
                    count="count"
                )
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.numeric_cols_:
            X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)

        report_rows = []
        for col in self.numeric_cols_ + self.categorical_cols_:
            missing_before = int(X[col].isna().sum())
            filled_model = filled_make = filled_global = 0
            numeric = col in self.numeric_cols_

            # 1. Model group
            if col in self.model_values_ and \
               {"Make", "Model_clean"}.issubset(X.columns):
                stats = self.model_values_[col]
                for idx in X.index[X[col].isna()]:
                    key = (X.at[idx, "Make"], X.at[idx, "Model_clean"])
                    if key not in stats.index:
                        continue
                    row = stats.loc[key]
                    if row["count"] < self.min_group_size:
                        continue
                    value = row["median" if numeric else "mode"]
                    if pd.notna(value):
                        X.at[idx, col] = value
                        filled_model += 1

            # 2. Make group
            if col in self.make_values_ and "Make" in X.columns:
                stats = self.make_values_[col]
                for idx in X.index[X[col].isna()]:
                    make = X.at[idx, "Make"]
                    if make not in stats.index:
                        continue
                    value = stats.loc[make]
                    if pd.notna(value):
                        X.at[idx, col] = value
                        filled_make += 1

            # 3. Global fallback
            fallback = self.global_values_.get(col, np.nan)
            if pd.notna(fallback):
                mask = X[col].isna()
                filled_global = int(mask.sum())
                X.loc[mask, col] = fallback

            report_rows.append({
                "Feature": col, "Missing_Before": missing_before,
                "Filled_Model_Group": filled_model,
                "Filled_Make_Group":  filled_make,
                "Filled_Global":      filled_global,
                "Missing_Final":      int(X[col].isna().sum())
            })

        self.report_ = pd.DataFrame(report_rows)
        if self.report:
            print("\nFINAL GROUP-BASED IMPUTATION REPORT")
            display(self.report_)
        return X

# ============================================================
# Yeo-Johnson transformer (works with negative values too)
# Normalizes numeric features without affecting the target
# ============================================================
class YeoJohnsonTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.numeric_columns_ = [
            c for c in X.select_dtypes(include=["number"]).columns
            if c != "Price"
        ]
        self.transformer_ = PowerTransformer(
            method="yeo-johnson", standardize=False
        )
        if self.numeric_columns_:
            self.transformer_.fit(X[self.numeric_columns_])
        return self

    def transform(self, X):
        X = X.copy()
        if self.numeric_columns_:
            X[self.numeric_columns_] = self.transformer_.transform(
                X[self.numeric_columns_]
            )
        return X

# ============================================================
# Scaler that fits on TRAIN only, applies to TEST
# (avoids data leakage)
# ============================================================
class NumericScaler(BaseEstimator, TransformerMixin):
    def __init__(self, exclude=('Price',), scaler=None):
        self.exclude = exclude
        self.scaler = scaler

    def fit(self, X, y=None):
        X = X.copy()
        cols = X.select_dtypes(include='number').columns.tolist()
        self.columns_ = [c for c in cols if c not in self.exclude]
        base = self.scaler if self.scaler is not None else RobustScaler()
        self.scaler_ = clone(base)
        if self.columns_:
            self.scaler_.fit(X[self.columns_])
        return self

    def transform(self, X):
        X = X.copy()
        if getattr(self, "columns_", None):
            X[self.columns_] = self.scaler_.transform(X[self.columns_])
        return X

# ============================================================
# One-Hot Encoder that handles unseen categories
# (handle_unknown='ignore' → unknowns become all-zeros)
# ============================================================
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols, drop_cols=None, encoder=None):
        self.cat_cols = cat_cols
        self.drop_cols = drop_cols or []
        self.encoder = encoder

    def fit(self, X, y=None):
        X = X.copy()
        for c in self.drop_cols:
            X = X.drop(columns=c, errors='ignore')
        self.cat_cols_ = [c for c in self.cat_cols if c in X.columns]
        base = self.encoder if self.encoder is not None else OneHotEncoder(
            drop='first', sparse_output=False, handle_unknown='ignore'
        )
        self.encoder_ = clone(base)
        if self.cat_cols_:
            self.encoder_.fit(X[self.cat_cols_])
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.drop_cols:
            X = X.drop(columns=c, errors='ignore')
        if getattr(self, "cat_cols_", None):
            enc = self.encoder_.transform(X[self.cat_cols_])
            enc_df = pd.DataFrame(
                enc,
                columns=self.encoder_.get_feature_names_out(self.cat_cols_),
                index=X.index
            )
            X = X.drop(columns=self.cat_cols_)
            X = pd.concat([X, enc_df], axis=1)
        return X

# ============================================================
# Preprocessing pipeline factory (row-preserving steps only)
#   A: keeps power_rpm / torque_rpm      B: drops them
# ============================================================
def make_preprocess(drop_rpm=False):
    steps = [
        ("cap_kilometer",       FunctionTransformer(cap_kilometer, validate=False)),
        ("feature_engineering", FunctionTransformer(feature_engineering, validate=False)),
        ("impute_from_model",   FunctionTransformer(impute_from_model, validate=False)),
        ("clean_model",         FunctionTransformer(clean_model, kw_args={"report": False},
                                                    validate=False)),
        ("model_make_imputer",  ModelMakeImputer(min_group_size=3, report=False)),
        ("final_group_imputer", GroupBasedImputer(min_group_size=3, report=False)),
        ("drop_original_cols",  FunctionTransformer(drop_original_columns, validate=False)),
    ]
    if drop_rpm:
        steps.append(("drop_rpm", FunctionTransformer(drop_rpm_features, validate=False)))
    return Pipeline(steps)


# ============================================================
# Stage-2 settings + FULL pipeline factory
#   raw data -> preprocess -> drop redundant cols -> Yeo-Johnson
#   -> scaler -> one-hot -> LinearRegression on log1p(Price)
# ============================================================
CAT_COLS = ['Make', 'Fuel Type', 'Transmission', 'Color',
            'Owner', 'Seller Type', 'Drivetrain']
DROP_COLS = ['Model_clean', 'Location']

def make_full_pipeline(drop_rpm=False):
    return Pipeline([
        ("preprocess",  make_preprocess(drop_rpm=drop_rpm)),
        ("select",      FunctionTransformer(drop_redundant_columns, validate=False)),
        ("yeo_johnson", YeoJohnsonTransformer()),
        ("scaler",      NumericScaler(scaler=StandardScaler())),
        ("encoder",     CategoricalEncoder(cat_cols=CAT_COLS, drop_cols=DROP_COLS)),
        ("model",       TransformedTargetRegressor(
                            regressor=LinearRegression(),
                            func=np.log1p, inverse_func=np.expm1)),
    ])

# ============================================================
# Extra features ON TOP of the existing preprocessing (row-wise, no leakage)
#   - log_km        : log of Kilometer (the raw column is very skewed)
#   - power_per_cc  : engine efficiency / "sportiness"
#   - Make_Base     : Make + first word of Model  (e.g. "bmw x1"),
#                     a middle level between Make and the ~870 distinct trims
# ============================================================
def add_extra_features(X):
    X = X.copy()
    if "Kilometer" in X.columns:
        X["log_km"] = np.log1p(X["Kilometer"])
    X["power_per_cc"] = X["power_bhp"] / X["engine_cc"].replace(0, np.nan)
    first = X["Model_clean"].astype("string").str.split().str[0].fillna("")
    X["Make_Base"] = (X["Make"].astype("string").str.lower() + " " + first).str.strip()
    return X

# ============================================================
# Improved pipeline (reuses YOUR make_preprocess unchanged)
#   raw -> make_preprocess -> extra features
#       -> [numeric as-is | target-encode high-cardinality | ordinal low-cardinality]
#       -> tree model on log1p(Price)
# Trees don't need scaling / Yeo-Johnson / dropping correlated columns,
# so we keep ALL numeric features here (Kilometer, engine_cc, torque_nm, Length...).
# TargetEncoder is cross-fitted inside fit() -> no target leakage.
# Model_clean and Location are USED here (the linear pipeline dropped them).
# ============================================================
TE_COLS  = ["Make", "Model_clean", "Make_Base", "Location"]
LOW_COLS = ["Fuel Type", "Transmission", "Owner", "Seller Type", "Drivetrain", "Color"]

def make_improved_pipeline(kind="hgb", drop_rpm=True):
    if kind == "rf":
        reg = RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                                    n_jobs=-1, random_state=42)
    else:
        reg = HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=600, max_leaf_nodes=15, min_samples_leaf=15,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=30, random_state=42)
    encode = ColumnTransformer([
        ("num", "passthrough", make_column_selector(dtype_include="number")),
        ("te",  TargetEncoder(target_type="continuous", cv=5), TE_COLS),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                               encoded_missing_value=-1), LOW_COLS),
    ])
    return Pipeline([
        ("preprocess", make_preprocess(drop_rpm=drop_rpm)),
        ("extra",      FunctionTransformer(add_extra_features, validate=False)),
        ("encode",     encode),
        ("model",      TransformedTargetRegressor(regressor=reg, func=np.log1p,
                                                  inverse_func=np.expm1)),
    ])

# ============================================================
# Metrics that are NOT dominated by a few luxury cars
#   R2_log     : R2 on log(price)  (same scale the model is trained on)
#   MedAPE_%   : median absolute % error  ("typical" error)
#   within_15% : share of cars predicted within +-15% of the real price
# R2_raw / MAE / RMSE are kept so you can compare with section 11.
# ============================================================
def full_report(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    ape = np.abs(y_pred - y_true) / y_true
    return {
        "R2_log":     r2_score(np.log1p(y_true), np.log1p(y_pred)),
        "R2_raw":     r2_score(y_true, y_pred),
        "MAE":        mean_absolute_error(y_true, y_pred),
        "RMSE":       np.sqrt(mean_squared_error(y_true, y_pred)),
        "MedAPE_%":   100 * np.median(ape),
        "within_15%": 100 * (ape <= 0.15).mean(),
    }
