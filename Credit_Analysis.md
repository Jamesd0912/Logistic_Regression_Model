# Logistic_Regression_Model
This is a logistic regression machine learning model and corresponding credit analysis on healthy vs. high-risk loans

# Credit Analysis Report

## Overview of the Analysis

The purpose of the following credit analysis was to train and evaluate machine learning models with imbalanced classes. The lending and credit industry is comprised of imbalanced data classes, as approved and healthy loans far outweigh high-risk and defaulted loans. Our goal in building the model is for the model to assess a credit analysis on future potential loans, to accurately predict high-risk loans before loan disbursement, and to limit losses on potentially high-risk loans. The information used to build and train the model was a dataset of historical lending activity of 77,537 clients, from a peer=to-peer lending service company. The dataset contained loan size, interest rate, borrower income, debt-to-income ratio, the number of borrowers' current credit accounts, the number of derogatory marks on borrowers' credit profiles, borrowers' total debt, and the current loan status.

The process of building our models first started by importing and reading our data into a pandas dataframe. We then sliced the dataframe to isolate the "loan status" from the original dataframe data as our target variable we called "y". We isolated this data point within the dataframe because "loan status" is classified with either  0 (healthy loan) or a 1, meaning high-risk loa. The remaining data points from the original dataframe we reclassified as a new dataframe under a new variable "X". Afterwards, we used the 'tain_test_split' function from the sklearn.model_selection library to create 4 separate variable datasets for our model to utilize, ('X_train', 'X_test', 'y_train', 'y_test'), with 75% of the datafame allocated for the two train variables and the other 25% for the two test variables. We then utilized the 'LogisticRegression' method on our lending datasets to create our model, which in turn, was used to call the '.fit' and '.predict' functions, the results of which we stored in a variable called "testing_predictions".

Since we are dealing with an imbalanced class, that being the credit and lending industry, we wanted to create a second model with a rebalanced dataset. To complete the second model we first needed to re-balance our data by creating more random instances of class 1 (high-risk loan) of "loan status", or variable "y_train" so that it equals instances of class 0 (healthy loan). This was done by utilizing the 'RandomOverSampler' module from the 'imblearn.over_sampling' library. Using our "X_train" and "y_train" dataset variables, created for our first model, as arguments for the '.fit_resample' function, we created two new dataset variables called "X_oversampled" and "y_oversampled". After resampling the data, we verified that we now had the same number of instances of classes within "loan status" by calling the '.value_counts' function on our "y_oversampled" variable, which both now contained 56,277 instances. Once our resampling was complete, we again used the 'LogisticRegression' method to create a new model, which we again used the model to fit and predict our "X_oversampled" and "y_oversampled" variables under the new variable "oversampled_prediction".



## Results

Machine learning model 1 is a logistic regression model using the original "X_train" and "y_train" datasets. Machine learning model 2 is also a logistic regression model using the new resampled "X_oversampled" and "y_oversampled" datasets.
The results of both models are as follows: 

* Machine Learning Model 1:
  * For Model 1, target 0 (healthy loan), we returned a perfect score of 1.0 for precision, recall & f1-score.
  * For Model 1, target 1 (high-risk loan), we returned a precision score of 0.87, a recall score of 0.89, and f1-score of 0.88.
  * Model 1 had a balanced accuracy score of 0.944.
  * Per the Classification report, the accuracy score is 0.99.



* Machine Learning Model 2:
  * For Model 2, target 0 (healthy loan), we also returned a perfect score of 1.0 for precision, recall & f1-score.
  * For Model 2, target 1 (high-risk loan), we returned a precision score of 0.87, a perfect recall score of 1.0, and f1-score of 0.93.
  * Model 1 had a balanced accuracy score of 0.995.
  * Per the Classification report, the accuracy score is 1.0.

## Summary

While both models performed excellent and either one would be sufficient, it does seem that the second model, with the oversampled data, performed slightly better than the original model. In regards to healthy loans, or target '0', both models returned a score of 1.0 for precision, recall and F-1. When it came to high-risk loans, or target '1', both models returned a precision score of 0.87 but the oversampled model (model 2) had a slightly better score for recall and f1-score metrics. The second model had a recall score of 1.00 and F-1 score of 0.93 verus model 1's recall of 0.89 and F-1 pf 0.88. Since the second model was better at predicting target '1`, the overall accuracy of the second model was 1.0 compared to the first model's accuracy score of 0.99. 

I would recommed the second model to all my potential clients. Since the prupose of this model is to predict high-risk loans and high-risk loans contain a larger emphasis, due to the risk of potential losses to a company, we would want to provide our clients with the best model available that can predict high-risk loans. While both models had perfect 1.0 scores across the board in predicting healthy loans, the only other variable is concerning high-risk loans. With the improved performance of model 2 in regards to predicting high-risk loans combined with the importance put on high-risk loans, model 2 is the obvious choice to reccomend. 


