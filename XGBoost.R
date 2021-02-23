# XGBoost R Tutorial
# https://xgboost.readthedocs.io/en/latest/R-package/xgboostPresentation.html#

library(data.table)
library(caTools)
library(xgboost)
library(caret)

credit_card_raw = fread("creditcard.csv")

# # 31 columns & 284807 rows
# dim(df)
# # V1~V28 all have mean 0 & no missing value
# summary(df)
# # 492 fraud & 284315 non-fraud
# table(df$Class)
# # 0.17% are fraud
# round(table(df$Class)[2]/nrow(df), 4)
# 
# # remove Time
# df[,Time:=NULL]
# # XGBoost only works with numeric vectors
# df$Class = as.numeric(df$Class)
# 
# # 80% as training set, 20% as test set
# set.seed(820)
# split = sample.split(df$Class, SplitRatio=0.8)
# train = as.matrix(subset(df, split==TRUE))
# test = as.matrix(subset(df, split==FALSE))
# 
# # 227846 rows in training set
# dim(train)
# # 56961 rows in test set
# dim(test)
# # same proportion of fraud as in df
# round(table(train[,"Class"])[2]/nrow(train), 4)
# round(table(test[,"Class"])[2]/nrow(test), 4)


# Create train and test dataset
credit_card_raw[, test:=0]
credit_card_raw[, "Time":= NULL]
credit_card_raw[sample(nrow(credit_card_raw), 284807*0.2), test:=1]
test <- credit_card_raw[test==1]
train <- credit_card_raw[test==0]
train[, "test" := NULL]
test[, "test" := NULL]
credit_card_raw[, "test" := NULL]

# Convert datatables to dataframes for downsampling
setDF(train)
setDF(test)

# Downsample
set.seed(1)
train$Class <- factor(train$Class)
downsample.train <- downSample(train[, -ncol(train)], train$Class)

test$Class <- factor(test$Class)
downsample.test <- downSample(test[, -ncol(test)], test$Class)


# train XGBoost model
xgb = xgboost(data=data.matrix(downsample.train[,1:29]), 
              label=as.numeric(downsample.train$Class)-1, 
              objective = "binary:logistic",
              max.depth = 2, eta = 1, nthread = 2, nrounds = 25)

# measure model performance on training set
pred = predict(xgb, data.matrix(downsample.train[,1:29]))
pred = as.numeric(pred>0.5)
print(head(pred))
training_accuracy = mean(pred==(as.numeric(downsample.train$Class)-1))
print(paste("Model accuracy on training set:", training_accuracy))

# make predictions
predictions = predict(xgb, data.matrix(downsample.test[,1:29]))
length(predictions) == dim(downsample.test)[1]

# transform predictions to binary results
predictions = as.numeric(predictions>0.5)
print(head(predictions))

# measure model performance on test set
test_accuracy = mean(predictions==(as.numeric(downsample.test$Class)-1))
print(paste("Model accuracy on test set:", test_accuracy))