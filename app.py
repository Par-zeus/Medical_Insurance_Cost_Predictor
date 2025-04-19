import streamlit as st
import pickle
import numpy as np

# Load model
with open('model.pkl', 'rb') as f:
    model, le_sex, le_smoker, le_region = pickle.load(f)

st.title("💸 Medical Insurance Cost Predictor")
st.write("Fill in the details below to predict your annual medical insurance cost.")

# User input
age = st.slider("Age", 18, 100)
sex = st.selectbox("Sex", le_sex.classes_)
bmi = st.number_input("BMI", 10.0, 50.0)
children = st.selectbox("Number of Children", list(range(6)))
smoker = st.selectbox("Smoker", le_smoker.classes_)
region = st.selectbox("Region", le_region.classes_)

# Predict button
if st.button("Predict Insurance Cost"):
    input_data = np.array([[age,
                            le_sex.transform([sex])[0],
                            bmi,
                            children,
                            le_smoker.transform([smoker])[0],
                            le_region.transform([region])[0]]])
    
    prediction = model.predict(input_data)
    st.success(f"Estimated Annual Insurance Cost: ₹{prediction[0]:,.2f}")
