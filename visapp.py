import pandas as pd
import joblib
import streamlit as st

# ---------------- LOAD FILES ----------------
model = joblib.load("rf_model.pkl")
country_encoder = joblib.load("country_encoder.pkl")
visa_encoder = joblib.load("visa_encoder.pkl")
model_columns = joblib.load("model_columns.pkl")


# ---------------- PREPROCESS ----------------
def preprocess_input(data):
    df = pd.DataFrame([data])

    df["application_date"] = pd.to_datetime(df["application_date"])
    df["month"] = df["application_date"].dt.month

    df["nationality"] = country_encoder.transform(df["nationality"])
    df["visa_status"] = visa_encoder.transform(df["visa_status"])

    df = df.drop(columns=["application_date"])

    df = pd.get_dummies(df)

    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[model_columns]

    return df


# ---------------- PREDICTION ----------------
def predict_processing_time(input_data):
    processed_data = preprocess_input(input_data)
    prediction = model.predict(processed_data)
    return round(prediction[0], 2)


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Visa Processing Predictor",
    page_icon="🌍",
    layout="centered"
)

# ---------------- CSS ----------------
st.markdown("""
<style>
/* Background */
body {
    background: linear-gradient(135deg, #667eea, #764ba2);
}

/* Title */
.title {
    text-align: center;
    font-size: 34px;
    font-weight: bold;
    color: white;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #e0e0e0;
    margin-bottom: 25px;
}

/* Card */
.card {
    background: rgba(255, 255, 255, 0.12);
    backdrop-filter: blur(12px);
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0px 8px 30px rgba(0,0,0,0.2);
}

/* Result Card */
.result {
    background: rgba(255,255,255,0.9);
    padding: 25px;
    border-radius: 16px;
    text-align: center;
    margin-top: 20px;
}

/* Result Number */
.result-number {
    font-size: 40px;
    font-weight: bold;
    color: #4CAF50;
}

/* Button */
.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 45px;
    font-size: 16px;
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border: none;
}

/* Input labels */
label {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">🌍 Visa Processing Time Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Estimate how long your visa might take</div>', unsafe_allow_html=True)

# ---------------- INPUT CARD ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    nationality = st.selectbox(
        "🌏 Nationality",
        ["India", "Brazil", "Mexico"]
    )

with col2:
    visa_status = st.selectbox(
        "📄 Visa Status",
        ["Approved", "Pending", "Refused", "Administrative Processing"]
    )

application_date = st.date_input("📅 Application Date")

predict_btn = st.button("🚀 Predict Processing Time")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- RESULT ----------------
if predict_btn:
    input_data = {
        "nationality": nationality,
        "visa_status": visa_status,
        "application_date": str(application_date)
    }

    result = predict_processing_time(input_data)

    st.markdown('<div class="result">', unsafe_allow_html=True)

    st.markdown("### ⏳ Estimated Processing Time")
    st.markdown(f'<div class="result-number">{result} days</div>', unsafe_allow_html=True)

    # Insight message
    if result < 30:
        st.success("⚡ Fast processing expected")
    elif result < 90:
        st.warning("⏱ Moderate processing time")
    else:
        st.error("🐢 May take longer than usual")

    st.markdown('</div>', unsafe_allow_html=True)