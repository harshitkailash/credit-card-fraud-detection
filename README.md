# credit-card-fraud-detection

## Context

Credit card fraud detection is a crucial task for financial institutions to prevent financial losses and protect customers. Anomalies or outliers in transaction data often indicate potential fraudulent activities. This project focuses on training and comparing two anomaly detection algorithms, Isolation Forest (IF) and Local Outlier Factor (LOF), to detect fraudulent credit card transactions. We will also deploy the best-performing model using Streamlit for real-time fraud detection.

## Dataset

The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, ... V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

## Step 1 : Data Preprocessing

1. _Data Loading and Inspection_ : Loaded the dataset and inspect the data for any missing values or anomalies.
   
2. _Data Visualization_ : Visualized the class distribution to understand the imbalance in the dataset.

3. _Splitting the Dataset_ : Separated the dataset into fraud and normal transactions for further analysis and preprocessing.

4. _Data Preprocessing_: Since the data is PCA transformed scaling is not required.

## Step 2: Training the Models

Now it is time to start building the model .The types of algorithms we are going to use to try to do anomaly detection on this dataset are as follows :

1. *Isolation Forest Algorithm :*

One of the newest techniques to detect anomalies is called Isolation Forests. The algorithm is based on the fact that anomalies are data points that are few and different. As a result of these properties, anomalies are susceptible to a mechanism called isolation.

This method is highly useful and is fundamentally different from all existing methods. It introduces the use of isolation as a more effective and efficient means to detect anomalies than the commonly used basic distance and density measures. Moreover, this method is an algorithm with a low linear time complexity and a small memory requirement. It builds a good performing model with a small number of trees using small sub-samples of fixed size, regardless of the size of a data set.

Typical machine learning methods tend to work better when the patterns they try to learn are balanced, meaning the same amount of good and bad behaviors are present in the dataset.

How Isolation Forests Work The Isolation Forest algorithm isolates observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature. The logic argument goes: isolating anomaly observations is easier because only a few conditions are needed to separate those cases from the normal observations. On the other hand, isolating normal observations require more conditions. Therefore, an anomaly score can be calculated as the number of conditions required to separate a given observation.

The way that the algorithm constructs the separation is by first creating isolation trees, or random decision trees. Then, the score is calculated as the path length to isolate the observation.

2. *Local Outlier Factor(LOF) Algorithm*

The LOF algorithm is an unsupervised outlier detection method which computes the local density deviation of a given data point with respect to its neighbors. It considers as outlier samples that have a substantially lower density than their neighbors.

The number of neighbors considered, (parameter n_neighbors) is typically chosen 1) greater than the minimum number of objects a cluster has to contain, so that other objects can be local outliers relative to this cluster, and 2) smaller than the maximum number of close by objects that can potentially be local outliers. In practice, such informations are generally not available, and taking n_neighbors=20 appears to work well in general.

**Model Initialization :**

We defined the contamination fraction based on the proportion of fraudulent transactions.

We initialized the Isolation Forest and Local Outlier Factor models with appropriate parameters.

## Step 3: Model Evaluation and Observations

**Model Evaluation**

1. For Local Outlier Factor, we used fit_predict to train the model and predict outliers in one step.

2. For Isolation Forest, we used fit to train the model and predict to predict outliers.

3. We convert the prediction results to 0 for valid transactions and 1 for fraudulent transactions.

4. We calculate and print the number of errors, accuracy score, and classification report for each model.

**Obervations**

1. Isolation Forest detected 689 errors versus Local Outlier Factor detecting 935 errors.

2. Isolation Forest has a 99.75 % more accurate than LOF of 99.67 %.

3. When comparing error precision & recall for 2 models , the Isolation Forest performed much better than the LOF as we can see that the detection of fraud cases is around 27 % versus LOF detection rate of just 2 %.

4. So overall Isolation Forest Method performed much better in determining the fraud cases which is around 30%.

# Application for Credit Card Fraud Detection

## Introduction

This documentation provides a comprehensive guide to deploying a pre-trained Isolation Forest model using Streamlit for credit card fraud detection. The application allows users to input transaction features and get real-time predictions on whether the transaction is legitimate or fraudulent.

## Code Explanation

The provided code defines and runs a Streamlit application with functionalities for user input, prediction, and result display.

**1. Importing Libraries** : 
The necessary libraries for the application include Streamlit for the web interface, pandas for data manipulation, and pickle for loading the pre-trained model.

**2. Loading the Pre-trained Model** :
Load the pre-trained Isolation Forest model from a pickle file.

**3. Defining the Streamlit Application** :
Define the main function that sets up the Streamlit application.

**4. Creating Input Fields** :
Create input fields for each transaction feature (V1 to V28) using st.number_input. This allows users to enter feature values.

**5. Submit Button and Prediction** :
Add a submit button that triggers the prediction when clicked. Convert the user input into a DataFrame, make a prediction using the pre-trained model, and display the result.

**6. Running the Streamlit Application** :
Execute the streamlit run command in the terminal to run the Streamlit application. Open a web browser and navigate to the local URL provided by Streamlit to interact with the application.
