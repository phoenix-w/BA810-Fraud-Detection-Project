library(data.table)
library("solitude")
library(dplyr)

dd <- fread("/Users/jeffrey/Documents/Boston University/BU-QST-Masters/Spring 2020/BA810/Team Project/Data/creditcard.csv")


# Create train and test dataset
dd[, test:=0]
dd[sample(nrow(dd), 142403), test:=1]
dd.test <- dd[test==1]
dd.train <- dd[test==0]
dd.train[, "test" := NULL]
dd.test[, "test" := NULL]

# Dataset to feed to iforest
for.test <- copy(dd.test)
for.train <- copy(dd.train)
for.train[, c("Time", "Class") := NULL]
for.test[, c("Time", "Class") := NULL]


# initiate an isolation forest
iso <- isolationForest$new(sample_size = length(for.train))
# fit for data
iso$fit(for.train)




# Obtain anomaly scores
scores_train = iso$predict(for.train)
scores_train[order(anomaly_score, decreasing = TRUE)]
dd.train$outlier <- as.factor(ifelse(scores_train$anomaly_score >=0.60, "outlier", "normal"))

scores_unseen = iso$predict(for.test)
scores_unseen[order(anomaly_score, decreasing = TRUE)]
dd.test$outlier <- as.factor(ifelse(scores_unseen$anomaly_score >=0.60, "outlier", "normal"))




# Train dataset True Positive and True Negative
train.fraud.accuracy <- sum(dd.train$Class[dd.train$outlier == "outlier"])/sum(dd.train$Class)
train.real.accuracy <- length(dd.train$Class[dd.train$Class == 0 & dd.train$outlier == "normal"])/sum(dd.train$Class == 0)

# Test dataset True Positive and True Negative
test.fraud.accuracy <- sum(dd.test$Class[dd.test$outlier == "outlier"])/sum(dd.test$Class)
test.real.accuracy <- length(dd.test$Class[dd.test$Class == 0 & dd.test$outlier == "normal"])/sum(dd.test$Class == 0)