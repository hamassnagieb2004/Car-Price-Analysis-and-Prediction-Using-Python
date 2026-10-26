# Used Car Price Prediction (India)

Predict the resale price of a used car from its specs, age, mileage, brand and model name.
Dataset: *Car details v4* (2,059 listings, 20 columns).

## Approach
| Step | What / why |
|---|---|
| Target | `log(Price)` – price is heavily right-skewed (skew 4.97 → 0.49) |
| Preprocessing | original pipeline kept: Kilometer cap, text parsing (cc/bhp/Nm), imputation from the Model string, then Make+Model and Make/global group imputers |
| Extra features | log_km, power_per_cc, Make_Base (brand + model family) |
| Encoding | cross-fitted **target encoding** for Make / Model / base-model / Location (no leakage), ordinal for the rest |
| Models | Linear Regression (original baseline) · Random Forest · Histogram Gradient Boosting |
| Validation | same 80/20 split (seed 42) as the original notebook + 5-fold CV; metrics that are robust to luxury outliers (R² on log, Median APE, % within 15%) |

## Run
```bash
pip install -r requirements.txt
python train.py --data "car details v4.csv"   # compares linear vs trees, saves outputs/
python predict.py                              # example prediction
```
Outputs: `outputs/metrics.csv`, `outputs/car_price_pipeline.joblib`.

## Results
Fill in from `outputs/metrics.csv` after running (before: Linear Regression, test R² 0.60 / MAE 364K on raw price).
