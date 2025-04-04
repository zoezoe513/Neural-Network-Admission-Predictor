import streamlit as st
import pandas as pd
from data_loader import load_data
from preprocessing import preprocess_data
from model import train_model
from evaluate import evaluate_model

st.set_page_config(page_title="Admission Predictor", page_icon="🎓")
st.title("🎓 UCLA Graduate School Admission Predictor")

# === Load and Preprocess ===
try:
    df = load_data()
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)
    model = train_model(X_train, y_train)
    acc, report = evaluate_model(model, X_test, y_test)
except Exception as e:
    st.error(f"❌ Error loading or preparing data: {e}")
    st.stop()

# === Input Form ===
st.subheader("📥 Enter Applicant Information")

input_data = {}
for feature in feature_names:
    if 'University_Rating' in feature or 'Research' in feature:
        continue  # skip for now (we handle below)
    elif feature in ['GRE_Score', 'TOEFL_Score']:
        input_data[feature] = st.slider(feature, min_value=260, max_value=340, value=300)
    elif feature == 'CGPA':
        input_data[feature] = st.slider(feature, min_value=0.0, max_value=10.0, value=8.0, step=0.1)
    elif feature in ['SOP', 'LOR']:
        input_data[feature] = st.slider(feature, min_value=1.0, max_value=5.0, value=3.0, step=0.5)

# === Categorical Inputs ===
unirating = st.selectbox("University Rating", [1, 2, 3, 4, 5])
research = st.radio("Research Experience", [0, 1])

# === Build Feature Vector ===
input_row = {col: 0 for col in feature_names}
for k, v in input_data.items():
    input_row[k] = v
input_row[f'University_Rating_{unirating}'] = 1
input_row[f'Research_{research}'] = 1

input_df = pd.DataFrame([input_row])
input_scaled = scaler.transform(input_df)

# === Prediction ===
if st.button("Predict Admission Chance"):
    try:
        pred = model.predict(input_scaled)[0]
        result = "✅ Likely to be Admitted" if pred == 1 else "❌ Less Likely to be Admitted"
        st.success(result)
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# === Model Performance ===
st.subheader("📊 Model Performance")

# Clean Accuracy
accuracy = report['accuracy'] if 'accuracy' in report else None
if accuracy is not None:
    st.write(f"**Accuracy:** `{accuracy:.2%}``")

# Prepare and display table
report_df = pd.DataFrame(report).transpose()
report_df = report_df.round(2)

# Drop accuracy from the table (already shown above)
if 'accuracy' in report_df.index:
    report_df = report_df.drop('accuracy')

# Display the classification report
st.dataframe(report_df.style.set_caption("Classification Report").format(precision=2))
