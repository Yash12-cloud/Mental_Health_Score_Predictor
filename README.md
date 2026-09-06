# 🧠 Student Social Media & Mental Health Score Predictor

A machine learning project that predicts a student's **Mental Health Score (0–10)** based on their social media usage habits and lifestyle. It ships with a **FastAPI prediction API** and a **Streamlit web UI**, and is deployable to [Render](https://render.com) in one click.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

## 📊 Model Performance

The final model is a **Random Forest Regressor** (tuned with grid search):

| Metric | Value |
|---|---|
| Test R² | **0.933** |
| Test MAE | 0.244 |
| Test RMSE | 0.345 |
| CV R² (5-fold) | 0.898 |

## 🗂️ Project Structure

```
Mental_Health_Score_Predictor/
├── app/
│   ├── app.py              # FastAPI backend (serves /predict, /meta, /countries)
│   └── streamlit_app.py    # Streamlit frontend (user-friendly form)
├── data/
│   ├── raw/                # Original Kaggle-style dataset (not committed)
│   └── processed/          # Cleaned dataset (not committed)
├── models/
│   ├── best_model.joblib   # Trained RandomForest pipeline (35 MB)
│   ├── best_model_params.json   # Hyperparameters + test metrics
│   └── train_ranges.json   # Min/max of numeric features seen in training
├── notebooks/
│   ├── cleaning.ipynb      # Data cleaning & preprocessing
│   ├── model_training.ipynb      # Baseline model comparison
│   └── hyperparameter_tuning.ipynb  # Grid search + final model
├── render.yaml             # Render deployment blueprint (2 services)
└── requirements.txt
```

## 🔮 Features Used for Prediction

| Feature | Type |
|---|---|
| `Age`, `Gender`, `Country`, `Academic_Level` | Demographics |
| `Most_Used_Platform`, `Purpose_Of_Use` | Social media profile |
| `Avg_Daily_Usage_Hours`, `Daily_Unlocks` | Usage intensity |
| `Study_Hours`, `Physical_Activity_Hours`, `Sleep_Hours_Per_Night` | Lifestyle |
| `Stress_Level` (Low / Medium / High / Very High) | Ordinal |

## 🚀 Run Locally

```bash
# 1. Create a virtual environment
python -m venv .myenv
source .myenv/bin/activate        # Windows: .myenv\Scripts\activate
pip install -r requirements.txt

# 2. Start the API (from the repo root)
uvicorn app.app:app --port 8000

# 3. In another terminal, start the UI
streamlit run app/streamlit_app.py
```

- API docs (Swagger): http://127.0.0.1:8000/docs
- Web UI: http://localhost:8501

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/countries` | List of supported countries |
| `GET` | `/meta` | All allowed categories + training ranges |
| `POST` | `/predict` | Predict the mental health score |

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 21,
    "Gender": "male",
    "Country": "india",
    "Academic_Level": "undergraduate",
    "Purpose_Of_Use": "education",
    "Most_Used_Platform": "instagram",
    "Avg_Daily_Usage_Hours": 5.0,
    "Daily_Unlocks": 120,
    "Study_Hours": 4.0,
    "Physical_Activity_Hours": 1.0,
    "Sleep_Hours_Per_Night": 7.0,
    "Stress_Level": "medium"
  }'
```

Example response:

```json
{
  "predicted_mental_health_score": 6.8,
  "normalized_input": { "Gender": "Male", "Country": "India", "...": "..." },
  "warnings": []
}
```

Inputs are auto-normalized (`"male"` → `"Male"`, `"high school"` → `"High School"`) and validated against the categories the model was trained on. If a number falls outside the training range, a warning is returned.

## ☁️ Deploy to Render

This repo includes a [`render.yaml`](render.yaml) blueprint that deploys **two free services**:

1. **mental-health-api** — the FastAPI backend (`uvicorn app.app:app`)
2. **mental-health-ui** — the Streamlit frontend, wired to the API via the `API_URL` env var

### One-click deploy

Click the **Deploy to Render** button at the top of this README, or:

1. Push this repo to your GitHub account
2. On [Render](https://dashboard.render.com) → **New → Blueprint** → select this repo
3. Render reads `render.yaml` and creates both services automatically

### Notes

- The UI service assumes the API lives at `https://mental-health-api.onrender.com`. If Render renames the service (name taken), update the `API_URL` env var of the **mental-health-ui** service in the Render dashboard.
- **Free tier caveat:** services spin down after ~15 minutes of inactivity. The first request after idle takes ~50 seconds (cold start). If the UI shows a connection error on first load, refresh once the API wakes up.
- The dataset CSVs are not committed (see `.gitignore`); the API reads its training ranges from `models/train_ranges.json`, so it runs without the data.

## 📓 Notebooks

1. `cleaning.ipynb` — load raw data, handle missing values/duplicates, encode targets
2. `model_training.ipynb` — preprocessing pipeline + baseline model comparison
3. `hyperparameter_tuning.ipynb` — grid search over Random Forest, final evaluation

## 📄 License

MIT
