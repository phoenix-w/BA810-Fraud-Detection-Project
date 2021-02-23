# XGBoost R Tutorial
# https://xgboost.readthedocs.io/en/latest/R-package/xgboostPresentation.html#

library(data.table)
library(caTools)
library(xgboost)
library(caret)

credit_card_raw = fread("creditcard.csv")

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


# Fit XGBoost model
xgb = xgboost(data=data.matrix(downsample.train[,1:29])
              ,label=as.numeric(downsample.train$Class)-1
              ,objective = "binary:logistic"
              ,max.depth = 2
              ,eta = 1
              ,nthread = 2
              ,nrounds = 25)

# Measure model performance on training set
pred = predict(xgb, data.matrix(downsample.train[,1:29]))
pred = as.numeric(pred>0.5)
print(head(pred))
training_accuracy = mean(pred==(as.numeric(downsample.train$Class)-1))
print(paste("Model accuracy on training set:", training_accuracy))

# Make predictions
predictions = predict(xgb, data.matrix(downsample.test[,1:29]))
length(predictions) == dim(downsample.test)[1]

# Transform predictions to binary results
predictions = as.numeric(predictions>0.5)
print(head(predictions))

# Measure model performance on test set
test_accuracy = mean(predictions==(as.numeric(downsample.test$Class)-1))
print(paste("Model accuracy on test set:", test_accuracy))


# Downsample the raw dataset: 492 frauds & 492 non-frauds
df = setDF(credit_card_raw)
df$Class = factor(df$Class)
downsample.df = downSample(df[,-ncol(df)], df$Class)

# XGBoost with cross-validation
xgb_cv = xgb.cv(data=data.matrix(downsample.df[,1:29])
                ,label=as.numeric(downsample.df$Class)-1
                ,objective = "binary:logistic"
                ,max.depth = 3
                ,eta = 1
                ,nthread = 2
                ,nrounds = 4
                ,nfold = 5
                ,metrics=list("rmse","auc"))

print(xgb_cv)