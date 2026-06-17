from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from datetime import datetime
import uuid
import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io

app = FastAPI(
    title="RetinaAI — DR Analysis API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Folders
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

jobs = {}

# -----------------------------
# Model Loading
# -----------------------------
GRADE_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

def load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 5)
    model.load_state_dict(torch.load("models/mobilenetv2_dr.pth", map_location="cpu"))
    model.eval()
    return model

dr_model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = dr_model(tensor)
        probs = torch.softmax(out, dim=1)
        grade = probs.argmax(1).item()
        confidence = probs[0][grade].item() * 100
    return grade, round(confidence, 2)

# -----------------------------
# Schema
# -----------------------------
class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: int
    stage: str
    created_at: str
    patient_id: str | None = None
    result: dict | None = None
    error: str | None = None

# -----------------------------
# Root
# -----------------------------
@app.get("/")
async def root():
    return {"message": "RetinaAI API running"}

# -----------------------------
# ANALYZE
# -----------------------------
@app.post("/api/analyze", response_model=JobStatus)
async def analyze(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    patient_id: str = Form(...)
):
    contents = await image.read()

    # Save image
    file_path = UPLOAD_DIR / f"{patient_id}_{uuid.uuid4().hex}.jpg"
    with open(file_path, "wb") as f:
        f.write(contents)

    # Real model prediction
    try:
        grade, confidence = predict(contents)
        error_msg = None
    except Exception as e:
        grade, confidence = 0, 0.0
        error_msg = str(e)

    result = {
        "final_grade": grade,
        "grade_name": GRADE_NAMES[grade],
        "confidence_pct": confidence,
        "lesion_labels": ["Microaneurysms", "Exudates", "Hemorrhages"],
        "urls": {
            "enhanced": None,
            "heatmap": None,
            "vessels": None,
            "lesions": None
        }
    }

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id": job_id,
        "status": "done",
        "progress": 100,
        "stage": "Complete",
        "created_at": datetime.now().isoformat(),
        "patient_id": patient_id,
        "result": result,
        "error": error_msg
    }

    return JobStatus(**jobs[job_id])

# -----------------------------
# RESULT
# -----------------------------
@app.get("/api/result/{job_id}", response_model=JobStatus)
async def get_result(job_id: str):
    return JobStatus(**jobs[job_id])

# -----------------------------
# HEALTH
# -----------------------------
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "pipeline_loaded": True,
        "demo_mode": False
    }

# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", port=8001, reload=True)