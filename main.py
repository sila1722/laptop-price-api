from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib, pandas as pd, os

PIPE_PATH = os.path.join("artifacts", "price_pipeline.joblib")
pipe = joblib.load(PIPE_PATH)

app = FastAPI(title="Laptop Price API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    Marka: str
    CPU: str
    GPU: str
    RAM_GB: float = Field(..., ge=1, le=1024)
    Depolama_GB: float = Field(..., ge=16, le=8192)
    Dokunmatik: float = Field(..., ge=0, le=1)  # 1=Var, 0=Yok
    Ekran_Inch: float = Field(..., ge=8, le=25)
    ResX: int = Field(..., ge=640, le=10000)
    ResY: int = Field(..., ge=480, le=10000)
    Hz: int = Field(..., ge=30, le=360)

class PredictResponse(BaseModel):
    predicted_price: float
    
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    df = pd.DataFrame([req.dict()])
    pred = float(pipe.predict(df)[0])
    return {"predicted_price": pred}
