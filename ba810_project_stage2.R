#load data
library(data.table)
#import the package
library(randomForest)
library(ggplot2)
library(grid)
library(gridExtra)
library(PRROC)
library(cvms)
library(pROC)
library(ROCR)
library(cvAUC)
setwd("~/Desktop/")
creditcard <- fread(file = "creditcard.csv")
#get structure
str(creditcard)
#get statistic summary for each column
summary(creditcard)        
#check missing values 
colSums(is.na(creditcard))   #no missing values 
#remove duplicated rows
cc = creditcard[!duplicated(creditcard), ]

cc$Class <- factor(cc$Class)
#Random Forest 
nrows <- nrow(cc)
set.seed(810)
index <- sample(1:nrow(cc), 0.7*nrows)
#train and validation set
train <- cc[index,]
val <- cc[-index,]

train_rf <- randomForest(train$Class~., data=train, ntree = 100, importance = TRUE)
print(train_rf)

varImpPlot(train_rf)

val$predicted<- predict(train_rf,val)

#AUC


  

