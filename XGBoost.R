# https://www.analyticsvidhya.com/blog/2016/01/xgboost-algorithm-easy-steps/
# https://www.kaggle.com/gpreda/credit-card-fraud-detection-using-xgboost
# https://www.kaggle.com/bonovandoo/fraud-detection-with-smote-and-xgboost-in-r

library(data.table)
library(xgboost)
library(readr)
library(stringr)
library(caret)
library(car)

df = fread("creditcard.csv")
