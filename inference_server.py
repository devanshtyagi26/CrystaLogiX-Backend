# inference_server.py
from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as rt
import numpy as np

app = FastAPI()

clf = rt.InferenceSession("models/stage1_classifier.onnx")
reg = rt.InferenceSession("models/stage2_regressor.onnx")

class PredictRequest(BaseModel):
    features: list[float]

@app.get("/")
def read_root():
    return {"message": "Welcome to the Material Property Prediction API. Use POST /predict to get predictions."}

@app.post("/api/predict")
def predict(req: PredictRequest):
    tensor = np.array([req.features], dtype=np.float32)

    clf_out = clf.run(None, {"float_input": tensor})
    label   = int(clf_out[0][0])
    probs   = clf_out[1][0].tolist()

    if label == 0:
        return {
            "stage1": {"is_metal": True,  "class_label": 0, "prob_metal": probs[0], "prob_non_metal": probs[1]},
            "stage2": {"bandgap_ev": None, "bandgap_category": "metal"}
        }

    reg_out   = reg.run(None, {"float_input": tensor})
    bandgap   = float(reg_out[0][0][0])
    category  = "semiconductor" if bandgap < 3.0 else "insulator"

    return {
        "stage1": {"is_metal": False, "class_label": 1, "prob_metal": probs[0], "prob_non_metal": probs[1]},
        "stage2": {"bandgap_ev": round(bandgap, 4), "bandgap_category": category}
    }