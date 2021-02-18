library(data.table)
library(dplyr)
library(caret)
library(readr)

dd <- fread("/Users/jeffrey/Documents/Boston University/BU-QST-Masters/Spring 2020/BA810/Team Project/Data/creditcard.csv")


# Create train and test dataset
dd[, test:=0]
dd[sample(nrow(dd), 284807*0.2), test:=1]
dd.test <- dd[test==1]
dd.train <- dd[test==0]
dd.train[, "test" := NULL]
dd.test[, "test" := NULL]


setDF(dd.train)

set.seed(1)
dd.train$Class <- factor(dd.train$Class)
dd.test$Class <- factor(dd.test$Class)
for.train <- downSample(dd.train[, -ncol(dd.train)], 
                      dd.train$Class)

table(for.train$Class)


ctrl <- trainControl(method = "cv", number = 5)

model <- train(Class ~ ., data = for.train, method = "knn", trControl = ctrl)
model

prediction <- predict(model, dd.test)

confusionMatrix(prediction, dd.test$Class, positive = "1")
