#src/api/app.py
import io
import torch
import numpy as np
from PIL import Image

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from src.inference.pipeline import predict


app = FastAPI(
    title="Industrial Defect Detection API",
    version="1.0"
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -----------------------------
# PREDICT ENDPOINT
# -----------------------------
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    temp_path = "temp_api_image.png"
    image.save(temp_path)
    
    try:
        result = predict(temp_path)
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )
    
    # Исправленная строка — heatmap уже list
    response = {
        "label": int(result["final_label"]),
        "decision": result["decision"],
        "classifier_prob": result["classifier_prob"],
        "patchcore_score": result["patchcore_score"],
        "bbox": result.get("bbox"),
        "heatmap": result["heatmap"]        # ← Здесь НЕ должно быть .cpu()
    }
    return JSONResponse(content=response)
