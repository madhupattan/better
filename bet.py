import pandas as pd
import joblib
import streamlit as st

# ---------------- LOAD FILES ----------------
model = joblib.load("rf_model.pkl")
country_encoder = joblib.load("country_encoder.pkl")
visa_encoder = joblib.load("visa_encoder.pkl")
model_columns = joblib.load("model_columns.pkl")


# ---------------- PREPROCESS FUNCTION ----------------
def preprocess_input(data):
    df = pd.DataFrame([data])

    df["application_date"] = pd.to_datetime(df["application_date"])
    df["month"] = df["application_date"].dt.month

    df["nationality"] = country_encoder.transform(df["nationality"])
    df["visa_status"] = visa_encoder.transform(df["visa_status"])

    df = df.drop(columns=["application_date"])

    df = pd.get_dummies(df)

    # Align with training columns
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[model_columns]

    return df


# ---------------- PREDICTION FUNCTION ----------------
def predict_processing_time(input_data):
    processed_data = preprocess_input(input_data)
    prediction = model.predict(processed_data)
    return round(prediction[0], 2)


# ---------------- UI DESIGN ----------------
st.set_page_config(
    page_title="Visa Processing Time",
    page_icon="⌛",
    layout="centered"
)

# ----------- CUSTOM CSS -----------
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}

.main-title {
    text-align: center;
    font-size: 32px;
    font-weight: bold;
    color: #2c3e50;
}

.card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
    margin-top: 20px;
}

.result-card {
    background-color: #ecfdf5;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    margin-top: 20px;
}

.big-text {
    font-size: 28px;
    font-weight: bold;
    color: #16a34a;
}
</style>
""", unsafe_allow_html=True)

# ----------- TITLE -----------
st.markdown('<div class="main-title">⌛ Visa Processing Time Estimator</div>', unsafe_allow_html=True)

# ----------- INPUT CARD -----------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("Enter Application Details")

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


# ----------- OUTPUT CARD -----------
if predict_btn:
    input_data = {
        "nationality": nationality,
        "visa_status": visa_status,
        "application_date": str(application_date)
    }

    result = predict_processing_time(input_data)

    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    st.markdown("### Estimated Processing Time")
    st.markdown(f'<div class="big-text">{result} days</div>', unsafe_allow_html=True)

    # Interpretation
    if result < 30:
        st.success("⚡ Fast processing expected")
    elif result < 90:
        st.warning("⏱ Moderate processing time")
    else:
        st.error("🐢 Processing may take longer")

    st.markdown('</div>', unsafe_allow_html=True)

