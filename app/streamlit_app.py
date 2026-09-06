"""
Streamlit web page to test the Mental Health prediction API.

Run it:
    1. Start the API first:      cd app && uvicorn app:app --port 8000
    2. Then start this page:     cd app && streamlit run streamlit_app.py
"""

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="Mental Health Score Predictor", page_icon="🧠", layout="wide")
st.title("🧠 Student Social Media & Mental Health")
st.caption("Enter student details to predict their Mental Health Score (0-10 scale).")


# ---------- Get allowed values from the API (single source of truth) ----------

@st.cache_data(ttl=60)
def get_meta():
    return requests.get(f"{API_URL}/meta", timeout=5).json()


try:
    meta = get_meta()
except requests.exceptions.ConnectionError:
    st.error(f"Cannot reach the API at `{API_URL}`. Is it running?")
    st.stop()

categories = meta["categories"]
training_ranges = meta["training_ranges"]


# ---------- Input form ----------

with st.form("student_form"):
    st.subheader("Student details")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 15, 30, 21)
        gender = st.selectbox("Gender", categories["Gender"])
        country = st.selectbox("Country", sorted(categories["Country"]))

    with col2:
        academic_level = st.selectbox("Academic Level", categories["Academic_Level"])
        purpose = st.selectbox("Purpose Of Use", categories["Purpose_Of_Use"])
        platform = st.selectbox("Most Used Platform", categories["Most_Used_Platform"])

    with col3:
        stress = st.select_slider("Stress Level", categories["Stress_Level"])
        sleep_hours = st.slider("Sleep Hours Per Night", 0.0, 24.0, 7.0, 0.1)
        physical_activity = st.slider("Physical Activity Hours", 0.0, 24.0, 1.5, 0.1)

    st.divider()
    col4, col5, col6 = st.columns(3)
    with col4:
        avg_usage = st.slider("Avg Daily Usage Hours", 0.0, 24.0, 4.0, 0.1)
    with col5:
        daily_unlocks = st.slider("Daily Unlocks", 0, 500, 130)
    with col6:
        study_hours = st.slider("Study Hours", 0.0, 24.0, 4.0, 0.1)

    submitted = st.form_submit_button("🔮 Predict Mental Health Score", use_container_width=True, type="primary")


# ---------- Call the API and show the result ----------

if submitted:
    payload = {
        "Age": age,
        "Gender": gender,
        "Country": country,
        "Academic_Level": academic_level,
        "Purpose_Of_Use": purpose,
        "Most_Used_Platform": platform,
        "Avg_Daily_Usage_Hours": avg_usage,
        "Daily_Unlocks": daily_unlocks,
        "Study_Hours": study_hours,
        "Physical_Activity_Hours": physical_activity,
        "Sleep_Hours_Per_Night": sleep_hours,
        "Stress_Level": stress,
    }

    with st.spinner("Predicting..."):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
        except requests.exceptions.ConnectionError:
            st.error("Lost connection to the API. Is uvicorn still running?")
            st.stop()

    if response.status_code == 422:
        # Show the API's validation errors (e.g. invalid category)
        for err in response.json()["detail"]:
            st.error(f"{err['loc'][-1]}: {err['msg']}")
    elif response.status_code == 200:
        result = response.json()
        score = result["predicted_mental_health_score"]

        st.divider()
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Predicted Mental Health Score", f"{score} / 10")
        with m2:
            if score >= 7:
                st.success("Good mental well-being 🙂")
            elif score >= 5:
                st.warning("Moderate mental well-being 😐")
            else:
                st.error("Poor mental well-being ⚠️")
        with m3:
            if result.get("warnings"):
                for w in result["warnings"]:
                    st.caption(f"⚠️ {w}")

        with st.expander("What was sent to the model?"):
            st.json(result)
    else:
        st.error(f"API error {response.status_code}: {response.text}")
