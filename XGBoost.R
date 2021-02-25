# XGBoost R Tutorial
# https://xgboost.readthedocs.io/en/latest/R-package/xgboostPresentation.html#

library(data.table)
library(caTools)
library(xgboost)
library(caret)
library(ROCR)
library(pROC)
library(ROSE)
library(randomForest)
library(ggplot2)

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

###=============================================================###
# Cross-validation (downsample.train)
dtrain = data.matrix(downsample.train[,1:29])
best_param = list()
best_seednumber = 1234
best_auc = Inf
best_auc_index = 0

for (iter in 1:10) {
  param <- list(objective = "binary:logistic", eval_metric = "auc",
                max_depth = 2, eta = 1, nthread = 2
  )
  cv.nround = 1000
  cv.nfold = 5
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  mdcv <- xgb.cv(data=dtrain, params = param,
                 nfold=cv.nfold, nrounds=cv.nround, verbose=0,
                 early_stopping_rounds=8, maximize=TRUE,
                 label=as.numeric(downsample.train$Class)-1)
  
  min_auc = min(mdcv$evaluation_log[, test_auc_mean])
  min_auc_index = which.min(mdcv$evaluation_log[, test_auc_mean])
  
  if (min_auc < best_auc) {
    best_auc = min_auc
    best_auc_index = min_auc_index
    best_seednumber = seed.number
    best_param = param
  }
}

nround = best_auc
set.seed(best_seednumber)


###=============================================================###
# Fit XGBoost model on downsampled training set
xgb = xgboost(data = dtrain,
              params = best_param, 
              nround = nround, 
              label=as.numeric(downsample.train$Class)-1)

# Apply XGBoost model on downsampled test set
predictions = predict(xgb, data.matrix(downsample.test[,1:29]))
# Transform predictions to binary results
predictions = as.numeric(predictions>0.5)
# Confusion matrix
cm1 = confusionMatrix(as.factor(predictions), downsample.test$Class
                      ,dnn=c("Prediction", "Reference"))
print(cm1)
# Plot ROC curve
roc1 = roc.curve(downsample.test$Class, as.factor(predictions), plotit = TRUE)
print(paste("Area under the curve (AUC):", round(roc1$auc, digits=3)))


# Apply XGBoost model on imbalanced test set
predictions2 = predict(xgb, data.matrix(test[,1:29]))
predictions2 = as.numeric(predictions2>0.5)
# Confusion matrix
cm2 = confusionMatrix(as.factor(predictions2), test$Class
                      ,dnn=c("Prediction", "Reference"))
print(cm2)
# Plot ROC curve
roc2 = roc.curve(test$Class, as.factor(predictions2), plotit = TRUE)
print(paste("Area under the curve (AUC):", round(roc2$auc, digits=3)))


###=============================================================###
# Cross-validation (imbalanced training set)
dtrain2 = data.matrix(train[,1:29])
best_param2 = list()
best_seednumber2 = 1234
best_auc2 = Inf
best_auc_index2 = 0

for (iter in 1:10) {
  param <- list(objective = "binary:logistic", eval_metric = "auc",
                max_depth = 2, eta = 1, nthread = 2
  )
  cv.nround = 1000
  cv.nfold = 5
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  mdcv <- xgb.cv(data=dtrain2, params = param,
                 nfold=cv.nfold, nrounds=cv.nround, verbose=0,
                 early_stopping_rounds=8, maximize=TRUE,
                 label=as.numeric(train$Class)-1)
  
  min_auc = min(mdcv$evaluation_log[, test_auc_mean])
  min_auc_index = which.min(mdcv$evaluation_log[, test_auc_mean])
  
  if (min_auc < best_auc2) {
    best_auc2 = min_auc
    best_auc_index2 = min_auc_index
    best_seednumber2 = seed.number
    best_param2 = param
  }
}

nround2 = best_auc2
set.seed(best_seednumber2)


###=============================================================###
# Fit XGBoost model on imbalanced training set
xgb2 = xgboost(data = dtrain2,
               params = best_param2, 
               nround = nround2, 
               label=as.numeric(train$Class)-1)


# Apply XGBoost model on downsampled test set
pred = predict(xgb2, data.matrix(downsample.test[,1:29]))
pred = as.numeric(pred>0.5)
# Confusion matrix
cm3 = confusionMatrix(as.factor(pred), downsample.test$Class
                      ,dnn=c("Prediction", "Reference"))
print(cm3)
# Plot ROC curve
roc3 = roc.curve(downsample.test$Class, as.factor(pred), plotit = TRUE)
print(paste("Area under the curve (AUC):", round(roc3$auc, digits=3)))


# Apply XGBoost model on imbalanced test set
pred2 = predict(xgb2, data.matrix(test[,1:29]))
pred2 = as.numeric(pred2>0.5)
# Confusion matrix
cm4 = confusionMatrix(as.factor(pred2), test$Class
                      ,dnn=c("Prediction", "Reference"))
print(cm4)
# Plot ROC curve
roc4 = roc.curve(test$Class, as.factor(pred2), plotit = TRUE)
print(paste("Area under the curve (AUC):", round(roc4$auc, digits=3)))


###=============================================================###
# Model comparison
print(paste("Sensitivity:"
            , cm1$byClass["Sensitivity"]
            ,"Specificity:"
            , cm1$byClass["Specificity"]
            , " AUC:"
            , round(roc1$auc, digits=3)
            , "(downsampled training & downsampled test)"
))
print(paste("Sensitivity:"
            , cm2$byClass["Sensitivity"]
            ,"Specificity:"
            , cm2$byClass["Specificity"]
            , " AUC:"
            , round(roc2$auc, digits=3)
            ,"(downsampled training & imbalanced test)"
))
print(paste("Sensitivity:"
            , cm3$byClass["Sensitivity"]
            ,"Specificity:"
            , cm3$byClass["Specificity"]
            , " AUC:"
            , round(roc3$auc, digits=3)
            ,"(imbalanced training & downsampled test)"
))
print(paste("Sensitivity:"
            , cm4$byClass["Sensitivity"]
            ,"Specificity:"
            , cm4$byClass["Specificity"]
            , " AUC:"
            , round(roc4$auc, digits=3)
            ,"(imbalanced training & imbalanced test)"
))