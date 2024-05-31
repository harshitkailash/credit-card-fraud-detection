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

For Local Outlier Factor, we used fit_predict to train the model and predict outliers in one step.

For Isolation Forest, we used fit to train the model and predict to predict outliers.

We convert the prediction results to 0 for valid transactions and 1 for fraudulent transactions.

We calculate and print the number of errors, accuracy score, and classification report for each model.

**Obervations**

Isolation Forest detected 689 errors versus Local Outlier Factor detecting 935 errors.

Isolation Forest has a 99.75 % more accurate than LOF of 99.67 %.

When comparing error precision & recall for 2 models , the Isolation Forest performed much better than the LOF as we can see that the detection of fraud cases is around 27 % versus LOF detection rate of just 2 %.

So overall Isolation Forest Method performed much better in determining the fraud cases which is around 30%.
