# Module-12-Homework-Supervised-Learning-
## Credit Risk Classification
### Overview
The purpose of this analysis is to train and evaluate machine learning models to identify the creditworthiness of borrowers using historical lending data. The dataset contains information on loans and their status (healthy or high-risk) from a peer-to-peer lending services company. Since this is an imbalanced classification problem, we'll use various techniques to improve model performance and evaluate the effectiveness of each technique.

### Data Preprocessing
Before we start building our models, we need to preprocess the data to get it into a format that we can work with. Here are the steps we took:

Loaded the lending_data.csv file into a Pandas DataFrame.
Created the labels set (y) from the “loan_status” column and the features (X) DataFrame from the remaining columns.
Checked the balance of the labels variable (y) using the value_counts function.
Split the data into training and testing datasets using train_test_split.

##### Logistic Regression Model with the Original Data
We first fit a logistic regression model with the original data. Here are the steps we took:

Fit a logistic regression model using the training data (X_train and y_train).
Predicted the testing data labels using the testing feature data (X_test) and the fitted model.
Evaluated the model's performance by calculating the accuracy score, generating a confusion matrix, and printing the classification report.
Answered the following question: How well does the logistic regression model predict both the 0 (healthy loan) and 1 (high-risk loan) labels?

##### Logistic Regression Model with Resampled Training Data.
Next, we used resampled data to train and evaluate a logistic regression model. Here are the steps we took:

Resampled the training data using RandomOverSampler from the imbalanced-learn library.
Fitted the LogisticRegression classifier using the resampled data and made predictions.
Evaluated the model's performance by calculating the accuracy score, generating a confusion matrix, and printing the classification report.
Answered the following question: How well does the logistic regression model, fit with oversampled data, predict both the 0 (healthy loan) and 1 (high-risk loan) labels?

#### Results
The following are the results of our analysis for each model:

##### Logistic Regression Model with the Original Data
Accuracy Score: 0.952
Confusion Matrix:
Overview
The purpose of this analysis is to train and evaluate machine learning models to identify the creditworthiness of borrowers using historical lending data. The dataset contains information on loans and their status (healthy or high-risk) from a peer-to-peer lending services company. Since this is an imbalanced classification problem, we'll use various techniques to improve model performance and evaluate the effectiveness of each technique.

Data Preprocessing
Before we start building our models, we need to preprocess the data to get it into a format that we can work with. Here are the steps we took:

Loaded the lending_data.csv file into a Pandas DataFrame.
Created the labels set (y) from the “loan_status” column and the features (X) DataFrame from the remaining columns.
Checked the balance of the labels variable (y) using the value_counts function.
Split the data into training and testing datasets using train_test_split.
Logistic Regression Model with the Original Data
We first fit a logistic regression model with the original data. Here are the steps we took:

Fit a logistic regression model using the training data (X_train and y_train).
Predicted the testing data labels using the testing feature data (X_test) and the fitted model.
Evaluated the model's performance by calculating the accuracy score, generating a confusion matrix, and printing the classification report.
Answered the following question: How well does the logistic regression model predict both the 0 (healthy loan) and 1 (high-risk loan) labels?
Logistic Regression Model with Resampled Training Data
Next, we used resampled data to train and evaluate a logistic regression model. Here are the steps we took:

Resampled the training data using RandomOverSampler from the imbalanced-learn library.
Fitted the LogisticRegression classifier using the resampled data and made predictions.
Evaluated the model's performance by calculating the accuracy score, generating a confusion matrix, and printing the classification report.
Answered the following question: How well does the logistic regression model, fit with oversampled data, predict both the 0 (healthy loan) and 1 (high-risk loan) labels?

#### Results
The following are the results of our analysis for each model:

##### Logistic Regression Model with Original Data
Balanced accuracy score: 95%
Precision and recall scores:
Healthy loans (0): precision = 1.00, recall = 0.99
High-risk loans (1): precision = 0.85, recall = 0.91

##### Logistic Regression Model with Resampled Data
Balanced accuracy score: 99%
Precision and recall scores:
Healthy loans (0): precision = 0.99, recall = 0.99
High-risk loans (1): precision = 0.99, recall = 0.99

### Summary
The logistic regression model trained with the original data had a high accuracy score of 95%, but its precision and recall scores for high-risk loans (1) were lower than those of the model trained with resampled data. On the other hand, the resampled data model had a balanced accuracy score of 99% and high precision and recall scores for both healthy and high-risk loans.

Based on these results, it is recommended to use the logistic regression model trained on resampled data using RandomOverSampler. This is because it is better suited to handling imbalanced data and produced higher performance metrics than the model trained on the original data.
