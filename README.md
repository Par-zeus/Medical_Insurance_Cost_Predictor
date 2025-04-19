# Medical_Insurance_Cost_Predictor
A simple and interactive Streamlit web application that predicts medical insurance costs based on user inputs like age, sex, BMI, number of children, smoking status, and region. This project uses Linear Regression from scikit-learn and is trained on the popular Insurance Dataset.

🚀 Features
Predicts annual medical insurance charges based on inputs.

Interactive web interface built with Streamlit.

Categorical features like sex, smoker, and region are encoded using LabelEncoder.

Easy-to-use sliders and dropdowns for user input.

Lightweight and fast to run locally.

🧠 Tech Stack
Frontend: Streamlit

Backend / ML: Python, scikit-learn, pandas, numpy

Model: Linear Regression

How To Run:

# Install dependencies
pip install -r requirements.txt

# Train the model (creates model.pkl)
python train_model.py

# Run the app
streamlit run app.py
