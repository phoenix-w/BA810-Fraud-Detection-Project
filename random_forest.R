#import the package
library(data.table)
library(randomForest)
library(ggplot2)
library(caret)
library(ROCR)
#load data
setwd("~/Desktop/")
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
pd <- predict(fit_rndfor, train[,-ncol(train)])
table(observed = train[,ncol(train)], predicted = pd)

pd.test <- predict(fit_rndfor, test[,-ncol(test)])
table(observed = test[,ncol(test)], predicted = pd.test)

#ROC curve
prediction_for_roc_curve <- predict(fit_rndfor,test[,-ncol(test)],type="prob")
pretty_colours <- c("#F8766D","#00BA38")
classes <- levels(test$Class)

for (i in 1:2)
{
  # Define which observations belong to class[i]
  true_values <- ifelse(test[,ncol(test)]==classes[i],1,0)
  # Assess the performance of classifier for class[i]
  pred <- prediction(prediction_for_roc_curve[,i],true_values)
  perf <- performance(pred, "tpr", "fpr")
  if (i==1)
  {
    plot(perf,main="ROC Curve",col=pretty_colours[i]) 
  }
  else
  {
    plot(perf,main="ROC Curve",col=pretty_colours[i],add=TRUE) 
  }
  # Calculate the AUC and print it to screen
  auc.perf <- performance(pred, measure = "auc")
  print(auc.perf@y.values)
}

#feature engineering - cross validation 
tst <- rfcv(trainx = test[,-ncol(test)], trainy = test[,ncol(test)], scale = "log")
tst$error.cv

