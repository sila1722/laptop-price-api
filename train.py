# train.py (final) — CPU/GPU drop + model-bazli doldurma + Pylance-clean
import argparse, os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from joblib import dump
from typing import Any, Dict, Optional

from sklearn.model_selection import train_test_split, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

# ---------------- Metrics ----------------
def metrics_report(y_true, y_pred) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)     # squared param yok -> eski sklearn uyumlu
    rmse = float(np.sqrt(mse))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


# ---------------- Feature types ----------------
def detect_feature_types(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])

    # Tamamen NaN olan ve constant kolonlari at
    non_all_nan = [c for c in X.columns if X[c].notna().sum() > 0]
    X = X[non_all_nan]
    non_constant = [c for c in X.columns if X[c].nunique(dropna=True) > 1]
    X = X[non_constant]

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    return X.columns.tolist(), num_cols, cat_cols


# ---------------- Preprocess ----------------
def build_preprocessor(num_cols, cat_cols):
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="constant", fill_value="Bilinmiyor")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), cat_cols),
        ],
        remainder="drop"
    )


# ---------------- Models ----------------
def get_models(random_state=42) -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "HistGBR": HistGradientBoostingRegressor(random_state=random_state),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, n_jobs=-1, random_state=random_state
        ),
    }
   
    return models


# ---------------- CV ----------------
def kfold_score(pipe: Pipeline, X: pd.DataFrame, y: pd.Series,
                n_splits=5, random_state=42) -> Dict[str, float]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rmses, maes, r2s = [], [], []
    for tr_idx, va_idx in kf.split(X):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_va)
        m = metrics_report(y_va, pred)
        rmses.append(m["RMSE"]); maes.append(m["MAE"]); r2s.append(m["R2"])
    return {"RMSE": float(np.mean(rmses)),
            "MAE": float(np.mean(maes)),
            "R2" : float(np.mean(r2s))}


# ---------------- Domain-aware imputation ----------------
def fill_missing_values(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """CPU/GPU bos ise satiri at; diger kolonlari model bazinda doldur."""
    # Kritik kolonlar: CPU, GPU -> bos ise drop
    for critical in ["CPU", "GPU"]:
        if critical in df.columns:
            df = df[df[critical].notna()]

    # Grup anahtari: name varsa name; yoksa Marka+Seri; o da yoksa Marka
    if "name" in df.columns:
        group_cols = ["name"]
    elif "Marka" in df.columns and "Seri" in df.columns:
        group_cols = ["Marka", "Seri"]
    else:
        group_cols = ["Marka"] if "Marka" in df.columns else []

    # Grup bazli doldurma
    if group_cols:
        for col in df.columns:
            if col == target_col: 
                continue
            if df[col].isna().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df.groupby(group_cols)[col].transform(
                        lambda x: x.fillna(x.median())
                    )
                else:
                    df[col] = df.groupby(group_cols)[col].transform(
                        lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else "Bilinmiyor")
                    )

    # Kalan NaN'ler: global median / "Bilinmiyor"
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna("Bilinmiyor")

    return df


# ---------------- Main ----------------
def main(args):
    csv_path   = Path(args.csv)
    target_col = args.target
    out_path   = Path(args.out)

    assert csv_path.exists(), f"CSV bulunamadi: {csv_path}"
    df = pd.read_csv(csv_path)
    assert target_col in df.columns, f"Target sutunu yok: {target_col}"

    # 1) Domain-aware doldurma
    df = fill_missing_values(df, target_col)

    # 2) Feature ayrimi
    all_X_cols, num_cols, cat_cols = detect_feature_types(df, target_col)
    if len(all_X_cols) == 0:
        raise RuntimeError("Kullanilabilir ozellik kalmadi. CSV kolonlarini kontrol et.")

    X = df[all_X_cols].copy()
    y = pd.to_numeric(df[target_col], errors="coerce")
    if y.isna().all():
        raise RuntimeError("Target sayiya cevrilemedi.")
    y = y.astype(float)

    print("Satir:", len(df))
    print("Ozellik:", len(all_X_cols), "| Sayisal:", len(num_cols), "| Kategorik:", len(cat_cols))

    # 3) Split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocess = build_preprocessor(num_cols, cat_cols)
    models = get_models()

    # 4) CV ve en iyi model secimi
    results: list[tuple[str, Dict[str, float]]] = []
    best_name: Optional[str] = None
    best_pipe: Optional[Pipeline] = None
    best_cv: Dict[str, float] | None = None

    print("\n=== 5-fold CV ===")
    for name, model in models.items():
        pipe = Pipeline([("prep", preprocess), ("model", model)])
        cv = kfold_score(pipe, X_tr, y_tr, n_splits=5, random_state=42)
        results.append((name, cv))
        print(f"{name:12s} | RMSE={cv['RMSE']:.2f} | MAE={cv['MAE']:.2f} | R^2={cv['R2']:.4f}")
        if best_cv is None or cv["RMSE"] < best_cv["RMSE"]:
            best_name, best_pipe, best_cv = name, pipe, cv

    assert best_cv is not None and best_pipe is not None and best_name is not None

    print(f"\nEn iyi model: {best_name} (CV RMSE={best_cv['RMSE']:.2f}, R^2={best_cv['R2']:.4f})")

    # 5) Final egitim + test
    best_pipe.fit(X_tr, y_tr)
    pred = best_pipe.predict(X_te)
    test_m: Dict[str, float] = metrics_report(y_te, pred)
    print(f"Test | RMSE={test_m['RMSE']:.2f} | MAE={test_m['MAE']:.2f} | R^2={test_m['R2']:.4f}")

    # 6) UI icin schema (choices + ranges)
    choices = {
        c: sorted(pd.Series(df[c].dropna().astype(str)).unique().tolist())
        for c in cat_cols
    }
    ranges: Dict[str, Dict[str, float]] = {}
    for c in num_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        mn = float(np.nanmin(s)) if np.isfinite(np.nanmin(s)) else 0.0
        mx = float(np.nanmax(s)) if np.isfinite(np.nanmax(s)) else 0.0
        ranges[c] = {"min": mn, "max": mx}

    # 7) Kaydet
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dump(best_pipe, out_path)
    meta = {
        "best_model": best_name,
        "cv": results,
        "test": test_m,
        "n_features_input": len(all_X_cols),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "all_cols": list(all_X_cols),
        "choices": choices,
        "ranges": ranges
    }
    with open(out_path.parent / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nKaydedildi: {out_path}")
    print(f"Meta: {out_path.parent / 'train_meta.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV yolu")
    ap.add_argument("--target", default="price", help="Hedef sutun adi")
    ap.add_argument("--out", default=os.path.join("artifacts", "price_pipeline.joblib"),
                    help="Kaydedilecek model yolu")
    args = ap.parse_args()
    main(args)
