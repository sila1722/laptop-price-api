# main.py (full)
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import joblib, pandas as pd, os, json

# ---------------- Paths ----------------
ART_DIR   = os.path.join("artifacts")
PIPE_PATH = os.path.join(ART_DIR, "price_pipeline.joblib")
META_PATH = os.path.join(ART_DIR, "train_meta.json")

# ---------------- Load model ----------------
# Not: /reload endpoint'i ile tekrar yükleyebilirsin
pipe = joblib.load(PIPE_PATH)

def load_meta() -> Dict[str, Any]:
    if not os.path.exists(META_PATH):
        return {}
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------- FastAPI ----------------
app = FastAPI(title="Laptop Price API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://price-prediction-website.vercel.app",   # Vercel prod
        # "http://localhost:3000",                       # Lokal gelistirme icin istersen ac
        # buraya gerekirse diger preview domainlerini ekle
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Helpers ----------------
def build_row_from_payload(payload: Dict[str, Any], meta: Dict[str, Any]) -> pd.DataFrame:
    """
    Frontend'den gelen payload'u egitimdeki kolonlara hizalar.
    - Eksik kategorik -> 'Bilinmiyor'
    - Eksik sayisal   -> ranges[min,max] ortalamasi (yoksa 0.0)
    - Ekstra alanlar  -> yok sayilir
    """
    num_cols = meta.get("num_cols", [])
    cat_cols = meta.get("cat_cols", [])
    all_cols = meta.get("all_cols", [])
    ranges   = meta.get("ranges", {})

    if not all_cols:
        raise ValueError("Meta bos. Once egitimi calistirip train_meta.json olustur.")

    row: Dict[str, Any] = {}
    for col in all_cols:
        if col in payload and payload[col] not in (None, ""):
            val = payload[col]
            if col in num_cols:
                try:
                    val = float(val)
                except Exception:
                    r = ranges.get(col, {})
                    val = float((r.get("min", 0.0) + r.get("max", 0.0)) / 2.0) if r else 0.0
            else:
                val = str(val)
            row[col] = val
        else:
            if col in num_cols:
                r = ranges.get(col, {})
                row[col] = float((r.get("min", 0.0) + r.get("max", 0.0)) / 2.0) if r else 0.0
            else:
                row[col] = "Bilinmiyor"

    return pd.DataFrame([row], columns=all_cols)

# ---------------- Endpoints ----------------
@app.get("/")
def root():
    return {"ok": True}

@app.get("/schema")
def get_schema():
    meta = load_meta()  # her istekte taze oku
    if not meta:
        return {"error": "train_meta.json not found. Run training first."}
    return {
        "num_cols": meta.get("num_cols", []),
        "cat_cols": meta.get("cat_cols", []),
        "choices": meta.get("choices", {}),
        "ranges": meta.get("ranges", {}),
        "all_cols": meta.get("all_cols", []),
    }

@app.post("/predict")
def predict(payload: Dict[str, Any] = Body(...)):
    meta = load_meta()  # her istekte taze oku
    df_one = build_row_from_payload(payload, meta)
    y_pred = float(pipe.predict(df_one)[0])
    return {"prediction": y_pred}

@app.post("/reload")
def reload_artifacts():
    """
    Opsiyonel: egitimden sonra backend'i komple yeniden baslatmadan modeli yeniden yuklemek icin.
    """
    global pipe
    pipe = joblib.load(PIPE_PATH)
    return {"ok": True, "reloaded": True}
