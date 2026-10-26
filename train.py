"""python train.py --data "car details v4.csv"
Same steps as the notebook: split -> row cleaning -> compare linear (original) vs tree models -> save best."""
import argparse
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split

from preprocessing import (make_full_pipeline, make_improved_pipeline, full_report,
                           prepare_rows, drop_high_kilometer)

def main(a):
    df = pd.read_csv(a.data)
    X, y = df.drop(columns="Price"), df["Price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    X_train, y_train = prepare_rows(X_train, y_train)
    X_train, y_train = drop_high_kilometer(X_train, y_train, threshold=500_000)

    cv = KFold(5, shuffle=True, random_state=42)
    scorer = lambda e, X_, y_: r2_score(np.log1p(y_), np.log1p(e.predict(X_)))
    models = {"linear": make_full_pipeline(drop_rpm=True),
              "rf": make_improved_pipeline("rf"), "hgb": make_improved_pipeline("hgb")}
    rows = {}
    for name, p in models.items():
        cv_r2 = cross_validate(clone(p), X_train, y_train, cv=cv, scoring=scorer, n_jobs=-1)["test_score"].mean()
        p.fit(X_train, y_train)
        rows[name] = {**full_report(y_test, p.predict(X_test)), "CV_R2_log": cv_r2}
        print(name, {k: round(float(v), 3) for k, v in rows[name].items()})
    res = pd.DataFrame(rows).T.round(3)
    out = Path(a.out); out.mkdir(exist_ok=True); res.to_csv(out / "metrics.csv")
    best = res["R2_log"].idxmax()
    final = clone(models[best]).fit(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))
    joblib.dump(final, out / "car_price_pipeline.joblib")
    print(f"Best: {best} -> saved {out/'car_price_pipeline.joblib'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="car details v4.csv"); ap.add_argument("--out", default="outputs")
    main(ap.parse_args())
