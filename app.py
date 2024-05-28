import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('isolation_forest_model.pkl')

# Streamlit app
st.title("Credit Card Fraud Detection Model")
st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

# Create input fields for user to enter feature values
feature_names = [f'V{i}' for i in range(1, 29)]
input_features = []
for feature in feature_names:
    value = st.text_input(f'Enter {feature}', '0')  # default value set to '0'
    input_features.append(value)

# Create a button to submit input and get prediction
if st.button("Submit"):
    try:
        # Get input feature values and convert to float
        features = np.array(input_features, dtype=np.float64).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)
        
        # Display result
        if prediction[0] == 1:
            st.write("Legitimate transaction")
        else:
            st.write("Fraudulent transaction")
    except ValueError as e:
        st.write("Please enter valid numerical values for all features.")

