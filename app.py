import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Liver Disease Classification", layout="centered")


# Load Artifacts

@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.joblib")
    scaler = joblib.load("scaler.joblib")
    le = joblib.load("label_encoder.joblib")
    return model, scaler, le

model, scaler, le = load_artifacts()


# Preprocessing + Prediction

def preprocess(df):
    # Drop target column if present
    if "category" in df.columns:
        df = df.drop(columns=["category"])

    # Ensure numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # Fill missing values
    df = df.fillna(df.median())

    # Apply scaling
    scaled = scaler.transform(df)
    return scaled

def predict(df):
    X_scaled = preprocess(df)
    preds = model.predict(X_scaled)
    labels = le.inverse_transform(preds)
    return labels


# UI Title

st.title("Liver Disease Prediction App")

st.write("Predict liver disease categories using laboratory features.")


# Prediction Form

st.subheader("🔹 Patient Prediction")

with st.form("single_input_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=32)
        sex = st.number_input("Sex (1 = male, 0 = female)", min_value=0, max_value=1, value=1)
        albumin = st.number_input("Albumin", value=40.0)
        alkaline_phosphatase = st.number_input("Alkaline Phosphatase", value=60.0)
        alanine_aminotransferase = st.number_input("ALT", value=20.0)

    with col2:
        aspartate_aminotransferase = st.number_input("AST", value=25.0)
        bilirubin = st.number_input("Bilirubin", value=8.0)
        cholinesterase = st.number_input("Cholinesterase", value=8.0)
        cholesterol = st.number_input("Cholesterol", value=4.0)
        creatinina = st.number_input("Creatinina", value=80.0)
        gamma_glutamyl_transferase = st.number_input("GGT", value=20.0)
        protein = st.number_input("Protein", value=70.0)

    submit = st.form_submit_button("Predict")

if submit:
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "albumin": albumin,
        "alkaline_phosphatase": alkaline_phosphatase,
        "alanine_aminotransferase": alanine_aminotransferase,
        "aspartate_aminotransferase": aspartate_aminotransferase,
        "bilirubin": bilirubin,
        "cholinesterase": cholinesterase,
        "cholesterol": cholesterol,
        "creatinina": creatinina,
        "gamma_glutamyl_transferase": gamma_glutamyl_transferase ,
        "protein": protein
    }])

    result = predict(input_df)[0]
    st.success(f"### 🟢 Predicted Category: **{result}**")