
library(data.table)
library(caret)
credit_card_raw  <-  fread("creditcard.csv")

# Create train and test credit_card_rawset
credit_card_raw[, test:=0]
credit_card_raw[, "Time":= NULL]
credit_card_raw[sample(nrow(credit_card_raw), 284807*0.2), test:=1]
test <- credit_card_raw[test==1]
train <- credit_card_raw[test==0]
train[, "test" := NULL]
test[, "test" := NULL]
credit_card_raw[, "test" := NULL]


# Convert credit_card_rawtables to credit_card_rawframes for downsampling
setDF(train)
setDF(test)

# Downsample
set.seed(1)
train$Class <- factor(train$Class)
downsample.train <- downSample(train[, -ncol(train)], train$Class)

test$Class <- factor(test$Class)
downsample.test <- downSample(test[, -ncol(test)], test$Class)



#build decision tree
#apply 5-folds cross validation to find the best parameter cp for decision tree
ctrl <- trainControl(method = "cv", number = 5)

model <- train(Class ~ ., data = downsample.train,
               method = "rpart",
               trControl = ctrl)

model

#find best cp for decision model

#the best model is about cp = 0.015

#evaluate the best model using test data

pred <- predict(model, downsample.test)

#performances 
confusionMatrix(pred, downsample.test$Class, positive = "1")






























