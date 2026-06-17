from fastapi import FastAPI, UploadFile, File
import shutil

from sr_model import enhance_image
from classifier import predict_stage

app = FastAPI()


labels = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}


recommendations = {
    0: "Maintain healthy lifestyle and regular eye checkups.",
    1: "Monitor condition and consult doctor if symptoms increase.",
    2: "Consult ophthalmologist soon for treatment.",
    3: "Urgent medical attention required.",
    4: "Immediate specialist treatment required."
}

@app.get("/")
def root():
    return {"status": "Backend is running"}

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):

    
    input_path = "input.png"
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    
    enhanced_path = enhance_image(input_path)

    
    stage, confidence = predict_stage(enhanced_path)

    
    label = labels[stage]
    recommendation = recommendations[stage]

    
    return {
        "stage": stage,
        "condition": label,
        "confidence": round(confidence, 2),
        "recommendation": recommendation
    }


