## Credit Card Fraud Detection

### Project Summary
As a team, we were tasked with finding a topic of our interest, locating a dataset, and applying supervised learning algorithms to solve the problem. We chose a credit card fraud dataset for the importance of the subject matter and the challenge brought by its imbalanced class distribution (492 frauds out of 284,807 transactions).

### Project Objective
We decided to build several supervised machine learning models and see which one performs the best in terms of identifying fraudulent transactions. Since the dataset is highly imbalanced with 99.8% of cases being non-fraudulent transactions, a simple measure like accuracy is not appropriate here as even a classifier which labels all transactions as non-fraudulent will have over 99% accuracy. The appropriate measure of model performance would be AUC (area under the ROC curve) and sensitivity (true positive rate).

### Data Source
Our dataset was taken from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud). It contains 284,807 rows and 31 columns. Each row represents a transaction made by credit cards in September 2013 by European cardholders. The majority of the features are nameless as a result of a PCA transformation. The only features which have not been transformed with PCA are "Time" and "Amount". Feature "Class" is the response variable and it takes value 1 in case of fraud and 0 otherwise.

*By Roozbeh Jafari, Jeffrey Leung, Phoenix Wang, Ying Wu, Yuzhe Zheng*
