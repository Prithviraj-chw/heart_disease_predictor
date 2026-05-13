# 🫀 Heart Disease Predictor

A machine learning web app that predicts the risk of heart disease based on clinical parameters. Built with **Streamlit** and **scikit-learn**, trained on the UCI Heart Disease dataset.

---

## 📸 Demo

> Fill in patient details → Click Predict → Get risk assessment with probability score

---

## 🧠 How It Works

1. Patient clinical data is entered via a Streamlit web interface
2. Categorical features are one-hot encoded to match training format
3. Numerical features are scaled using a pre-fitted StandardScaler
4. A K-Nearest Neighbors (KNN) classifier predicts heart disease risk
5. The result is displayed with a probability percentage

---

## 📊 Dataset

- **Source:** [UCI Heart Disease Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Size:** 918 patients
- **Target:** `HeartDisease` (0 = No, 1 = Yes)

### Features Used

| Feature | Description |
|---|---|
| Age | Age of the patient (years) |
| Sex | Biological sex (M/F) |
| ChestPainType | ASY, ATA, NAP, TA |
| RestingBP | Resting blood pressure (mm Hg) |
| Cholesterol | Serum cholesterol (mg/dL) |
| FastingBS | Fasting blood sugar > 120 mg/dL (0/1) |
| RestingECG | Resting ECG results (Normal/ST/LVH) |
| MaxHR | Maximum heart rate achieved |
| ExerciseAngina | Exercise-induced angina (Y/N) |
| Oldpeak | ST depression induced by exercise |
| ST_Slope | Slope of peak exercise ST segment (Up/Flat/Down) |

---

## 🤖 Model

- **Algorithm:** K-Nearest Neighbors (KNN)
- **Preprocessing:** StandardScaler on numerical features, One-Hot Encoding on categorical features
- **Train/Test Split:** 80/20
- **Test Accuracy:** ~87%

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/heart_disease_predictor.git
cd heart_disease_predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run heart_disease_pred.py
```

---

## 📁 Project Structure

```
heart_disease_predictor/
│
├── heart_disease_pred.py   # Streamlit web app
├── heart_pred.ipynb        # Model training notebook
├── knn.pkl                 # Trained KNN model
├── scaler.pkl              # Fitted StandardScaler
├── columns.pkl             # Expected feature columns
├── heart.csv               # Dataset (not included in repo)
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 📦 Requirements

```
streamlit
pandas
scikit-learn
joblib
numpy
```

---

## ⚠️ Disclaimer

This app is built for educational purposes only and is **not** intended for real medical diagnosis. Always consult a qualified healthcare professional for medical advice.

---

## 👨‍💻 Author

**Prithviraj** — built as a personal ML project to explore healthcare data and deploy predictive models with Streamlit.
