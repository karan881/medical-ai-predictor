from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List
from fastapi.middleware.cors import CORSMiddleware

from model import get_disease_matches


app = FastAPI(
    title="AI Medical Disease Predictor",
    description="Predict possible diseases based on symptoms using AI-powered matching with embeddings, fuzzy logic, and NLP.",
    version="1.0.0"
)

# ===============================
# CORS (Optional, useful for frontend JS or external use)
# ===============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (frontend apps, etc.)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# Static Files and Template Setup
# ===============================
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ===============================
# Request/Response Models
# ===============================
class SymptomsInput(BaseModel):
    symptoms: List[str] = Field(..., example=["fever", "headache", "stomach pain"])


class DiseaseResult(BaseModel):
    Disease: str
    Matched_Symptoms: List[str]
    Score: float
    Description: str
    Recommended_Drugs: str
    Test_Suggestions: str
    Specialist: str


class PredictionResponse(BaseModel):
    results: List[DiseaseResult]


# ===============================
# API Endpoint (JSON-based for programmatic use)
# ===============================
@app.post("/predict", response_model=PredictionResponse, summary="Predict Disease", tags=["Prediction"])
def predict_disease(input: SymptomsInput):
    if not input.symptoms:
        raise HTTPException(status_code=400, detail="Symptoms list cannot be empty.")
    try:
        result = get_disease_matches(input.symptoms)
        if not result:
            raise HTTPException(status_code=404, detail="No matching diseases found.")
        return {"results": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===============================
# Frontend: Home Page (Form-based)
# ===============================
@app.get("/", response_class=HTMLResponse)
@app.get("/home", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None, "error": None})


# ===============================
# Frontend: Form Handler
# ===============================
@app.post("/submit", response_class=HTMLResponse)
def handle_form(request: Request, symptoms: str = Form(...)):
    symptom_list = [sym.strip() for sym in symptoms.split(",") if sym.strip()]

    if not symptom_list:
        return templates.TemplateResponse("index.html", {"request": request, "error": "Please enter symptoms.", "result": None})

    try:
        result = get_disease_matches(symptom_list)
        if not result:
            return templates.TemplateResponse("index.html", {"request": request, "error": "No matching diseases found.", "result": None})
        return templates.TemplateResponse("index.html", {"request": request, "result": result, "error": None})
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "error": str(e), "result": None})
