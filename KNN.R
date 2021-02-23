library(data.table)
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

# Convert datatables to dataframes for downsampling
setDF(train)
setDF(test)

# Downsample
set.seed(1)
train$Class <- factor(train$Class)
downsample.train <- downSample(train[, -ncol(train)], train$Class)

test$Class <- factor(test$Class)
downsample.test <- downSample(test[, -ncol(test)], test$Class)


# Set cross validation parameters
knn_ctrl <- trainControl(method = "cv", number = 5)

# Run model
knn_model <- train(Class ~ ., data = downsample.train, method = "knn", trControl = knn_ctrl)


# Predict on downsampled test set
knn_prediction <- predict(knn_model, downsample.test)

# Confusion matrix
confusionMatrix(knn_prediction, downsample.test$Class, positive = "1")

# Predict on imbalanced test set 
prediction.imbalanced <- predict(knn_model, test)

# Confusion matrix
confusionMatrix(prediction.imbalanced, test$Class, positive = "1")


