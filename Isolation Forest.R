library(data.table)
library("solitude") # need to install
library(dplyr)
library(caret)

credit_card_raw <- fread("/Users/jeffrey/Documents/Boston University/BU-QST-Masters/Spring 2020/BA810/Team Project/Data/creditcard.csv")


# Create train and test dataset
credit_card_raw[, test:=0]
credit_card_raw[, "Time":= NULL]
credit_card_raw[sample(nrow(credit_card_raw), 284807*0.2), test:=1]
test <- credit_card_raw[test==1]
train <- credit_card_raw[test==0]
train[, "test" := NULL]
test[, "test" := NULL]
credit_card_raw[, "test" := NULL]

# Copy new data as to not disturb other models
iforest_train <- copy(train)
iforest_test <- copy(test)

# initiate an isolation forest
iso <- isolationForest$new(sample_size = length(iforest_train))
# fit for data
iso$fit(iforest_train)


# Obtain anomaly scores (According to documentation: "If the score is closer to 1 for a some observations, they are likely outliers. If the score for all observations hover around 0.5, there might not be outliers at all.")
iforest_scores_train = iso$predict(iforest_train)
iforest_scores_train[order(anomaly_score, decreasing = TRUE)]
iforest_train$predictions <- as.factor(ifelse(iforest_scores_train$anomaly_score >=0.60, 1, 0))

iforest_scores_test = iso$predict(iforest_test)
iforest_scores_test[order(anomaly_score, decreasing = TRUE)]
iforest_test$predictions <- as.factor(ifelse(iforest_scores_test$anomaly_score >=0.60, 1, 0))


# Train dataset True Positive and True Negative
train.fraud.accuracy <- sum(iforest_train$Class[iforest_train$outliers == "outlier"])/sum(iforest_train$Class)
train.real.accuracy <- length(iforest_train$Class[iforest_train$Class == 0 & iforest_train$outlier == "normal"])/sum(iforest_train$Class == 0)

confusionMatrix(iforest_train$predictions, as.factor(train$Class), positive = "1")

confusionMatrix(iforest_test$predictions, as.factor(test$Class), positive = "1")

# Test dataset True Positive and True Negative
test.fraud.accuracy <- sum(iforest_test$Class[iforest_test$outlier == "outlier"])/sum(iforest_test$Class)
test.real.accuracy <- length(iforest_test$Class[iforest_test$Class == 0 & iforest_test$outlier == "normal"])/sum(iforest_test$Class == 0)

roc.curve(test$Class, iforest_test$predictions, plotit = TRUE)

roc.curve(train$Class, iforest_train$predictions, plotit = TRUE)
