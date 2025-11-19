import shutil
import os
import zipfile
from fastapi import FastAPI, UploadFile, File
from config import load_config
from api.schemas import PredictResponse
from api.model_loader import init_model, model_service
from api.utils import load_and_preprocess_npy

app = FastAPI(title="LUNA25 Hybrid Model API")

CONFIG_PATH = "config/config.yaml"
cfg = load_config(CONFIG_PATH)

CHECKPOINT = "checkpoints/best.pth"

@app.on_event("startup")
def load_model():
    init_model(CHECKPOINT)

@app.post("/predict-file", response_model=PredictResponse)
async def predict_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".npy"):
        return {"file": file.filename, "probability": -1, "label": -1}

    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    x = load_and_preprocess_npy(temp_path, cfg)
    prob, label = model_service.predict_tensor(x)

    return PredictResponse(
        file=file.filename,
        probability=prob,
        label=label
    )


@app.post("/predict-folder")
async def predict_folder(zip_file: UploadFile = File(...)):
    zip_path = "/tmp/upload.zip"
    extract_dir = "/tmp/extracted"

    with open(zip_path, "wb") as f:
        shutil.copyfileobj(zip_file.file, f)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    results = []
    for fname in os.listdir(extract_dir):
        if fname.endswith(".npy"):
            path = os.path.join(extract_dir, fname)
            x = load_and_preprocess_npy(path, cfg)
            prob, label = model_service.predict_tensor(x)
            results.append({"file": fname, "probability": prob, "label": label})

    return {"results": results}
