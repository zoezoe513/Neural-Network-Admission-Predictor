# Admission Predictor (Neural Network)

This app has been built using Streamlit and deployed with Streamlit Community Cloud.

🔗 **Visit the app here**: [https://neural-network-admission-predictor-daevthsdr4hlhfeeujvbpd.streamlit.app/]  
🔐 **Password** (if needed): `streamlit`

---

## 📋 Description

This app predicts graduate school admission using a neural network based on GPA, GRE, TOEFL scores, and other academic indicators.

---

## 📁 Dataset

The model is trained on a structured dataset with features such as:
- GRE Score
- TOEFL Score
- CGPA
- SOP
- LOR
- University Rating
- Research Experience

---

## ⚙️ Technologies Used

- **Streamlit** – For building the interactive web application  
- **Scikit-learn** – For model training and evaluation  
- **Pandas & NumPy** – For data preprocessing and manipulation  
- **Matplotlib & Seaborn** – For data visualization and exploration (optional)

---

## 🤖 Model Summary

Uses an MLPClassifier from Scikit-learn with scaled features and one-hot encoded categorical variables. Outputs admission chances.

---

## 🚀 Future Enhancements

- Add support for additional datasets  
- Incorporate explainability tools like SHAP or LIME  
- Add charts to visualize user inputs and predictions

---

## 🧪 Local Installation

```bash
git clone https://github.com/zoezoe513/Neural-Network-Admission-Predictor
cd ucla_admission_predictor
python -m venv env
source env/bin/activate  # On Windows use `env\\Scripts\\activate`
pip install -r requirements.txt
streamlit run app.py
```

---

Thanks for using **Admission Predictor (Neural Network)**! 🙌  
Feel free to contribute or share your feedback.
