import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the saved models and accuracy data
with open("knn_model.pkl", "rb") as f:
    knn_model = pickle.load(f)

with open("dt_model.pkl", "rb") as f:
    dt_model = pickle.load(f)

with open("knn_accuracy.pkl", "rb") as f:
    knn_accuracy = pickle.load(f)

with open("dt_accuracy.pkl", "rb") as f:
    dt_accuracy = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# Display model information and accuracy
st.title("Air Quality Index (AQI) Prediction Models")

st.write("### KNN Model Accuracy")
st.write(f"KNN Model Accuracy: {knn_accuracy['knn_accuracy']:.2f}")

st.write("### Decision Tree Model Accuracy")
st.write(f"Decision Tree Model Accuracy: {dt_accuracy['dt_accuracy']:.2f}")

# User Input Section for Prediction
st.write("### Enter the features for prediction")

# Create input fields for all the features
input_values = []
for col in feature_columns:
    input_values.append(st.number_input(f"Enter {col}", value=0))

# When user submits, make predictions
if st.button("Predict"):
    # Convert the input values into a numpy array
    input_data = np.array(input_values).reshape(1, -1)
    
    # Make predictions using KNN and Decision Tree models
    knn_prediction = knn_model.predict(input_data)
    dt_prediction = dt_model.predict(input_data)

    # Show the results
    st.write("### Predictions:")
    st.write(f"KNN Model Prediction: {knn_prediction[0]}")
    st.write(f"Decision Tree Model Prediction: {dt_prediction[0]}")

    # Add interpretation for predictions
    if knn_prediction[0] == 0:
        st.write("KNN Prediction: Good")
    elif knn_prediction[0] == 1:
        st.write("KNN Prediction: Satisfactory")
    elif knn_prediction[0] == 2:
        st.write("KNN Prediction: Moderate")
    elif knn_prediction[0] == 3:
        st.write("KNN Prediction: Poor")
    elif knn_prediction[0] == 4:
        st.write("KNN Prediction: Very Poor")
    elif knn_prediction[0] == 5:
        st.write("KNN Prediction: Severe")

    if dt_prediction[0] == 0:
        st.write("Decision Tree Prediction: Good")
    elif dt_prediction[0] == 1:
        st.write("Decision Tree Prediction: Satisfactory")
    elif dt_prediction[0] == 2:
        st.write("Decision Tree Prediction: Moderate")
    elif dt_prediction[0] == 3:
        st.write("Decision Tree Prediction: Poor")
    elif dt_prediction[0] == 4:
        st.write("Decision Tree Prediction: Very Poor")
    elif dt_prediction[0] == 5:
        st.write("Decision Tree Prediction: Severe")
