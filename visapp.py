import pandas as pd
import joblib

# Load trained model
model = joblib.load("rf_model.pkl")

# Load encoders (if you saved them)
country_encoder = joblib.load("country_encoder.pkl")
visa_encoder = joblib.load("visa_encoder.pkl")

def preprocess_input(data):
    """
    Convert user input into model-ready format
    """

    df = pd.DataFrame([data])

    # Convert date
    df["application_date"] = pd.to_datetime(df["application_date"])

    # Feature Engineering
    df["month"] = df["application_date"].dt.month


    # Encode categorical variables
    df["nationality"] = country_encoder.transform(df["nationality"])
    df["visa_status"] = visa_encoder.transform(df["visa_status"])

    # Drop unused columns
    

    return df


def predict_processing_time(input_data):
    """
    Takes user input → returns predicted processing time
    """

    processed_data = preprocess_input(input_data)

    prediction = model.predict(processed_data)

    return round(prediction[0], 2)



#streamlit code- frontend

# pip install streamlit

#app.py - to run the streamlit app- streamlit run app.py

import streamlit as st
st.markdown(
    """
    <style>
    h1 {
        color: #2c3e50;
        text-align: center;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.set_page_config(page_title='Visa Processing Time Predictor',page_icon='⌛',layout='centered')
st.title('⌛ Visa Processing Time Estimator')


st.write("Enter application details below:")

# Input fields
col1,col2,col3=st.columns(3)
with col1:
    nationality= st.selectbox("🌏 Select Nationality", ["India", "Brazil", "Mexico"])

with col2:
    visa_status = st.selectbox("📄 Visa Status", ["Approved", "Pending", "Refused", 'Administrative Processing'])

with col3:
    application_date = st.date_input("Application Date") # date_input- allows teh user to pick the date

# Button
if st.button("Predict Processing Time"):

    input_data = {
        "nationality": nationality,
        "visa_status": visa_status,
        "application_date": str(application_date)
    }

    result = predict_processing_time(input_data)

    st.success(f"Estimated Processing Time: {result} days")


