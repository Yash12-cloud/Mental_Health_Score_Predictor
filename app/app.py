from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Annotated
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent  # repo root

# ---------- Load model and category lists once at startup ----------

model = joblib.load(BASE_DIR / "models" / "best_model.joblib")
prep = model.named_steps["prep"]

# The encoder already knows the allowed categories (from training).
CAT_NAMES = ["Gender", "Country", "Academic_Level", "Most_Used_Platform", "Purpose_Of_Use"]
cat_values = dict(zip(CAT_NAMES, prep.named_transformers_["cat"].categories_))
stress_values = list(prep.named_transformers_["ord"].categories_[0])

# Min/max values the model saw during training (used to warn about weird inputs)
NUM_COLS = ["Age", "Avg_Daily_Usage_Hours", "Daily_Unlocks", "Study_Hours",
            "Physical_Activity_Hours", "Sleep_Hours_Per_Night"]
import json

with open(BASE_DIR / "models" / "train_ranges.json") as f:
    train_ranges = {c: (v["min"], v["max"]) for c, v in json.load(f).items()}

app = FastAPI(
    title="Student Social Media & Mental Health API",
    description="Predicts Mental Health Score of a student based on social media usage and lifestyle features.",
    version="1.2.0",
)


class StudentData(BaseModel):
    Age: Annotated[int, Field(..., ge=15, le=30, description="Age of the Student")]
    Gender: Annotated[str, Field(..., description="Gender of the Student (Female or Male)")]
    Country: Annotated[str, Field(..., description="Country of the student, e.g. India, USA, Other. See GET /countries")]
    Academic_Level: Annotated[str, Field(..., description="Academic level of student (High School / Undergraduate / Graduate)")]
    Purpose_Of_Use: Annotated[str, Field(..., description="Purpose of use (Education / Entertainment / Networking / News)")]
    Most_Used_Platform: Annotated[str, Field(..., description="Most app used by student, e.g. Instagram")]
    Avg_Daily_Usage_Hours: Annotated[float, Field(..., ge=0, le=24, description="Daily usage of app in hrs")]
    Daily_Unlocks: Annotated[int, Field(..., ge=0, description="Daily unlocks of the apps")]
    Study_Hours: Annotated[float, Field(..., ge=0, le=24, description="Daily study hours")]
    Physical_Activity_Hours: Annotated[float, Field(..., ge=0, le=24, description="Daily physical activity in hours")]
    Sleep_Hours_Per_Night: Annotated[float, Field(..., gt=0, le=24, description="Daily sleeping hours")]
    Stress_Level: Annotated[str, Field(..., description="Stress level of student (Low / Medium / High / Very High)")]

    # Every text field is normalized: " male " -> "Male", "high school" -> "High School"
    @field_validator("Gender")
    @classmethod
    def check_gender(cls, v):
        return fix_category(v, cat_values["Gender"], "Gender")

    @field_validator("Country")
    @classmethod
    def check_country(cls, v):
        return fix_category(v, cat_values["Country"], "Country")

    @field_validator("Academic_Level")
    @classmethod
    def check_academic_level(cls, v):
        return fix_category(v, cat_values["Academic_Level"], "Academic_Level")

    @field_validator("Purpose_Of_Use")
    @classmethod
    def check_purpose(cls, v):
        return fix_category(v, cat_values["Purpose_Of_Use"], "Purpose_Of_Use")

    @field_validator("Most_Used_Platform")
    @classmethod
    def check_platform(cls, v):
        return fix_category(v, cat_values["Most_Used_Platform"], "Most_Used_Platform")

    @field_validator("Stress_Level")
    @classmethod
    def check_stress(cls, v):
        return fix_category(v, stress_values, "Stress_Level")


def fix_category(value, allowed, field_name):
    """Clean user input and match it (case-insensitive) to the trained category."""
    cleaned = " ".join(value.split())          # "  High   School " -> "High School"
    for option in allowed:
        if cleaned.lower() == option.lower():  # "male" == "Male"
            return option                      # return the correct form e.g. "Male"
    options = ", ".join(allowed)
    raise ValueError(f"Invalid {field_name} '{value}'. Allowed: {options}")


# ---------- Endpoints ----------

@app.get("/")
def home():
    return {"message": "Student Social Media & Mental Health Prediction API"}


@app.get("/countries")
def countries():
    return {"countries": sorted(cat_values["Country"])}


@app.get("/meta")
def meta():
    """All allowed values, for anyone building a frontend."""
    return {
        "categories": {k: sorted(v) for k, v in cat_values.items()} | {"Stress_Level": stress_values},
        "training_ranges": {k: {"min": v[0], "max": v[1]} for k, v in train_ranges.items()},
    }


@app.post("/predict")
def predict(data: StudentData):
    row = data.model_dump()  # dict of the (already cleaned) input

    # Warn if a number is outside what the model saw during training
    warnings = []
    for col, (low, high) in train_ranges.items():
        if row[col] < low or row[col] > high:
            warnings.append(f"{col}={row[col]} is outside the training range [{low}, {high}], prediction may be unreliable")

    try:
        df = pd.DataFrame([row])
        score = model.predict(df)[0]
        return {
            "predicted_mental_health_score": round(float(score), 2),
            "normalized_input": row,   # show the cleaned values we actually used
            "warnings": warnings,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
