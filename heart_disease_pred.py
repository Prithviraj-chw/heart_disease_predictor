import streamlit as st
import pandas as pd
import joblib

# Load model artifacts
model           = joblib.load("knn.pkl")
scaler          = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

#  Page config
st.set_page_config(page_title="Heart Disease Predictor", page_icon="🫀")
st.title("🫀 Heart Disease Predictor")
st.markdown("Fill in the patient details below and click **Predict**.")

# Input form
col1, col2 = st.columns(2)

with col1:
    age             = st.slider("Age", 18, 100, 40)
    sex             = st.selectbox("Sex", ["M", "F"])
    chest_pain      = st.selectbox("Chest Pain Type", ["ASY", "ATA", "NAP", "TA"])
    resting_bp      = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol     = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)

with col2:
    fasting_bs      = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
    resting_ecg     = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr          = st.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
    oldpeak         = st.slider("Oldpeak (ST Depression)", -3.0, 7.0, 0.0, step=0.1)
    st_slope        = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# Prediction
if st.button("Predict", use_container_width=True):

    # Start with all expected columns set to 0
    input_dict = {col: 0 for col in expected_columns}

    # Fill numerical features
    input_dict["Age"]        = age
    input_dict["RestingBP"]  = resting_bp
    input_dict["Cholesterol"] = cholesterol
    input_dict["FastingBS"]  = fasting_bs
    input_dict["MaxHR"]      = max_hr
    input_dict["Oldpeak"]    = oldpeak

    # One-hot encode categorical features
    if sex == "M":
        input_dict["Sex_M"] = 1

    if chest_pain == "ATA":
        input_dict["ChestPainType_ATA"] = 1
    elif chest_pain == "NAP":
        input_dict["ChestPainType_NAP"] = 1
    elif chest_pain == "TA":
        input_dict["ChestPainType_TA"] = 1
    # ASY → all three stay 0 (drop-first encoding)

    if resting_ecg == "Normal":
        input_dict["RestingECG_Normal"] = 1
    elif resting_ecg == "ST":
        input_dict["RestingECG_ST"] = 1
    # LVH → both stay 0

    if exercise_angina == "Y":
        input_dict["ExerciseAngina_Y"] = 1

    if st_slope == "Flat":
        input_dict["ST_Slope_Flat"] = 1
    elif st_slope == "Up":
        input_dict["ST_Slope_Up"] = 1
    # Down → both stay 0

    # Build DataFrame in the exact column order the model expects
    input_df     = pd.DataFrame([input_dict])[expected_columns]
    scaled_input = scaler.transform(input_df)
    prediction   = model.predict(scaled_input)[0]
    probability  = model.predict_proba(scaled_input)[0][1] * 100

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ **High Risk of Heart Disease** ({probability:.1f}% probability)")
    else:
        st.success(f"✅ **Low Risk of Heart Disease** ({probability:.1f}% probability)")