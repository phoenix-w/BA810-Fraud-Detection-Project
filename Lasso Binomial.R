library(data.table)
library(caTools)
library(xgboost)
library(caret)
library(ROCR)
library("pROC")
library("ROSE")
library(glmnet)
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

# Convert datatables to dataframes for downsampling
setDF(train)
setDF(test)

# Downsample
set.seed(1)
train$Class <- factor(train$Class)
downsample.train <- downSample(train[, -ncol(train)], train$Class)

test$Class <- factor(test$Class)
downsample.test <- downSample(test[, -ncol(test)], test$Class)


# Create formula
formula <- as.formula(Class ~ .)

# Downsample training set modeling 
downsample.train.matrix <- model.matrix(formula, downsample.train)[, -1]
y.downsample.train <- downsample.train$Class
downsample.fit <- cv.glmnet(downsample.train.matrix, y.downsample.train, family = "binomial", alpha = 1, nfolds = 10)

# Create testing matrices
downsample.test.matrix <- model.matrix(formula, downsample.test) [, -1]
imbalanced.test.matrix <- model.matrix(formula, test)[, -1]

# Predicting Downsample test data
downsample.test.predictions <- predict(downsample.fit, downsample.test.matrix, s = downsample.fit$lambda.min) 
predicted.classes <- ifelse(downsample.test.predictions > 0, 1, 0)
confusionMatrix(as.factor(predicted.classes), downsample.test$Class, positive = "1")
roc.curve(as.numeric(downsample.test$Class), as.numeric(predicted.classes), plotit = TRUE)

# Predicting imbalanced test data
test.predictions <- predict(downsample.fit, imbalanced.test.matrix, s = downsample.fit$lambda.min) 
predicted.classes <- ifelse(test.predictions > 0, 1, 0)
confusionMatrix(as.factor(predicted.classes), test$Class, positive = "1")
roc.curve(as.numeric(test$Class), as.numeric(predicted.classes), plotit = TRUE)

# Imbalanced training set modeling
class_0 = copy(train[train$Class == 0,])
x <- copy(class_0[sample(nrow(class_0), 4000),])
imbalanced.train <- rbind(x, train[train$Class == 1,])
train.matrix <- model.matrix(formula, imbalanced.train)[, -1]
y.train <- imbalanced.train$Class
imbalanced.fit <- cv.glmnet(train.matrix, y.train, family = "binomial", alpha = 1, nfolds = 10)

# Predicting Downsample test data
downsample.test.predictions <- predict(imbalanced.fit, downsample.test.matrix, s = imbalanced.fit$lambda.min) 
predicted.classes <- ifelse(downsample.test.predictions > 0, 1, 0)
confusionMatrix(as.factor(predicted.classes), downsample.test$Class, positive = "1")
roc.curve(as.numeric(downsample.test$Class), as.numeric(predicted.classes), plotit = TRUE)

#Predicting imbalanced test data
test.predictions <- predict(imbalanced.fit, imbalanced.test.matrix, s = imbalanced.fit$lambda.min) 
predicted.classes <- ifelse(test.predictions > 0, 1, 0)
confusionMatrix(as.factor(predicted.classes), test$Class, positive = "1")
roc.curve(as.numeric(test$Class), as.numeric(predicted.classes), plotit = TRUE)

