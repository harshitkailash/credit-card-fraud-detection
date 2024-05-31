import streamlit as st
import pandas as pd
import pickle

# Load the pre-trained Isolation Forest model
with open('isolation_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

def main():
    # Set the title and description
    st.title("Credit Card Fraud Detection Model")
    st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

    # Create input fields for user to enter feature values
    input_df = {}
    for i in range(1, 29):
        input_df[f'V{i}'] = st.number_input(f'V{i}', value=0.0)

    # Create a button to submit input and get prediction
    submit = st.button("Submit")

    if submit:
        # Convert input features to a DataFrame
        input_features = pd.DataFrame([input_df])

        # Make prediction
        prediction = model.predict(input_features)

        # Display result
        if prediction[0] == -1:
            st.write("Fraudulent transaction")
        else:
            st.write("Legitimate transaction")

# Run the Streamlit app
if __name__ == "__main__":
    main()
