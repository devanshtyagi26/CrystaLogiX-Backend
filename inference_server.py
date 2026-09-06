# inference_server.py
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
import onnxruntime as rt
import numpy as np
import os
from fastapi.middleware.cors import CORSMiddleware

# ── Config ────────────────────────────────────────────────────────────────────

EXPECTED_FEATURES = 86
API_KEY           = os.environ["API_SECRET_KEY"]  # hard fail if not set

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI()

# Only the Next.js server is allowed — not the browser directly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://crystalogix.devanshtyagi.app"],
    allow_methods=["POST"],
    allow_headers=["x-api-key", "Content-Type"],
)

# ── Models ────────────────────────────────────────────────────────────────────

try:
    clf = rt.InferenceSession("models/stage1_classifier.onnx")
    reg = rt.InferenceSession("models/stage2_regressor.onnx")
except Exception as e:
    raise RuntimeError(f"Failed to load ONNX models: {e}") from e

# ── Auth ──────────────────────────────────────────────────────────────────────

def verify_key(x_api_key: str = Header(default=None)):
    # Header(default=None) returns 401 instead of 422 when header is absent
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key.")

# ── Schema ────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    features: list[float]

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def read_root():
    return {
        "message": "Material Property Prediction API. POST /api/predict to run inference."
    }

@app.get("/api/health")
def health():
    return {"status": "ok", "models_loaded": clf is not None and reg is not None}

@app.post("/api/predict", dependencies=[Depends(verify_key)])
def predict(req: PredictRequest):
    if len(req.features) != EXPECTED_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {EXPECTED_FEATURES} features, got {len(req.features)}.",
        )

    tensor = np.array([req.features], dtype=np.float32)

    # ── Stage 1: classifier ───────────────────────────────────────────────
    clf_out = clf.run(None, {"float_input": tensor})
    label   = int(clf_out[0][0])
    probs   = clf_out[1][0].tolist()

    if label == 0:
        return {
            "stage1": {
                "is_metal":       True,
                "class_label":    0,
                "prob_metal":     round(probs[0], 6),
                "prob_non_metal": round(probs[1], 6),
            },
            "stage2": {
                "bandgap_ev":       None,
                "bandgap_category": "metal",
            },
        }

    # ── Stage 2: regressor (non-metals only) ─────────────────────────────
    reg_out  = reg.run(None, {"float_input": tensor})
    bandgap  = float(reg_out[0][0][0])
    category = "semiconductor" if bandgap < 3.0 else "insulator"

    return {
        "stage1": {
            "is_metal":       False,
            "class_label":    1,
            "prob_metal":     round(probs[0], 6),
            "prob_non_metal": round(probs[1], 6),
        },
        "stage2": {
            "bandgap_ev":       round(bandgap, 4),
            "bandgap_category": category,
        },
    }