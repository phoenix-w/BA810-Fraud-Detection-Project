#import the package
library(data.table)
library(randomForest)
library(ggplot2)
library(caret)
library(ROCR)
#load data
setwd("~/Documents/BA810")
credit_card_raw <- fread(file = "creditcard.csv")
#get structure
str(credit_card_raw)
#get statistic summary for each column
summary(credit_card_raw)        
#check missing values 
colSums(is.na(credit_card_raw))   #no missing values 
#remove duplicated rows
credit_card_raw = credit_card_raw[!duplicated(credit_card_raw), ]
#set Class as factor
credit_card_raw$Class <- factor(credit_card_raw$Class)
#Create train and test data set
credit_card_raw[, test:=0]
credit_card_raw[, "Time":= NULL]
credit_card_raw[sample(nrow(credit_card_raw), 283726*0.2), test:=1]
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

downsample.train <- downSample(train[, -ncol(train)], train$Class)

downsample.test <- downSample(test[, -ncol(test)], test$Class)

#Random Forest 

fit_rndfor <- randomForest(downsample.train$Class~., data=downsample.train, ntree = 500, importance = TRUE)

varImpPlot(fit_rndfor)

#make predictions 
pd <- predict(fit_rndfor, downsample.train[,-ncol(downsample.train)])
table(observed = downsample.train[,ncol(downsample.train)], predicted = pd)

pd.test <- predict(fit_rndfor, downsample.test[,-ncol(downsample.test)])
table(observed = downsample.test[,ncol(downsample.test)], predicted = pd.test)

#ROC curve
prediction_for_roc_curve <- predict(fit_rndfor,downsample.test[,-ncol(downsample.test)],type="prob")


pred <- prediction(prediction_for_roc_curve,true_values)
perf <- performance(pred, "tpr", "fpr")
  
plot(perf,main="ROC Curve", colorize=TRUE) 

# Calculate the AUC and print it to screen
auc.perf <- performance(pred, measure = "auc")
print(auc.perf@y.values)


#cross validation 
rf.cv <- rfcv(downsample.train[,-ncol(downsample.train)], downsample.train[,ncol(downsample.train)], cv.fold = 10) 
with(rf.cv, plot(n.var,error.cv))

#tuning parameter
bestmtry <- tuneRF(downsample.train[,-ncol(downsample.train)], downsample.train[,ncol(downsample.train)], stepFactor=1.5, improve=1e-5, ntree=500)
print(bestmtry) #mtry = 4 gives an accuracy of 94%; mtry: Number of variables randomly sampled as candidates at each split.

