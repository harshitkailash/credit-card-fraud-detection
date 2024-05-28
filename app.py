import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from pylab import rcParams
import streamlit as st
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]
data = pd.read_csv('creditcard.csv',sep=',')
## Getting the Fraud and the normal dataset 

fraud = data[data['Class']==1]
normal = data[data['Class']==0]
#Determining the number of fraud and valid transactions in the dataset

Fraud = data[data['Class']==1]

Valid = data[data['Class']==0]

outlier_fraction = len(Fraud)/float(len(Valid))
# Separate features and labels
X = data.drop('Class', axis=1)  # Features
y = data['Class']  # Labels
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Isolation Forest model
model = IsolationForest(n_estimators=100, max_samples=len(X), 
                                       contamination=outlier_fraction, verbose=0)
model.fit(X_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Convert predictions to binary (1: normal, -1: anomaly)
y_pred_binary = [1 if x == 1 else 0 for x in y_pred]

# Evaluate the model
print(confusion_matrix(y_test, y_pred_binary))
print(classification_report(y_test, y_pred_binary))

# Define the Streamlit app
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
        if prediction[0] == 1:
            st.write("Legitimate transaction")
        else:
            st.write("Fraudulent transaction")

# Run the Streamlit app
if __name__ == "__main__":
    main()

