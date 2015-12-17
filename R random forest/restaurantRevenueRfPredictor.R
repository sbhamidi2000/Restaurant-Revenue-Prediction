library(mice)
#library(randomForest)
library(Boruta)
library(caret)
library(Metrics)
library(gbm)
library(lubridate)

train <- read.csv("train.csv")
test  <- read.csv("test.csv")
pvals <- read.csv("PvalsImpSampling.csv")

n.train <- nrow(train)

test$revenue <- 1

myData <- rbind(train, test)
rm(train, test)

#Tranform Time
train$Open.Date <- as.POSIXlt("01/01/2015", format="%m/%d/%Y") - as.POSIXlt(train$Open.Date, format="%m/%d/%Y")
train$Open.Date <- as.numeric(train$Open.Date / 1000) #Scale for factors

#Consolidate Cities
train$City                                      <- as.character(train$City)
train$City[train$City.Group == "Other"]        <- "Other"
train$City[train$City == unique(train$City)[4]] <- unique(train$City)[2]
train$City                                      <- as.factor(train$City)
train$City.Group                                <- NULL

#Consolidate Types
train$Type <- as.character(train$Type)
train$Type[train$Type=="DT"] <- "IL"
train$Type[train$Type=="MB"] <- "FC"
train$Type <- as.factor(train$Type)

#Log Transform P Variables and Revenue
myData[, paste("P", 1:37, sep="")] <- as.numeric(0.5-(log(1 +myData[, paste("P", 1:37, sep="")])))

#Note: 1 column consolidated, col index changes
myData[5:41] <- lapply(myData[5:41],as.factor)
myData$revenue <- log(myData$revenue)

#Impute missing values
Pvals.train <- myData[1:n.train,5:41]
Pvals.train[Pvals.train==0] <- NA
imputedPvals <- mice(Pvals.train, seed=1234,meth='polyreg')
myData[1:n.train,18:41] <- complete(imputedPvals)

# Feature selection
important <- Boruta(revenue~., data=myData[1:n.train,],ntree=1000,mtry=5)

#Random Forest training model
model <- train(revenue~., 
               data=myData[1:n.train, c(important$finalDecision != "Rejected", TRUE)],'rf',ntree=1000,tuneGrid = data.frame(mtry = 5))

#Make a Prediction
prediction <- predict(model, myData[1:n.train, ])
cost <- (1/2*n.train)*sum((prediction - myData$revenue[1:n.train]))

#Make Submission
submit<-as.data.frame(cbind(seq(0, length(prediction) - 1, by=1), exp(prediction)))
colnames(submit)<-c("Id","Prediction")
write.csv(submit,"submission.csv",row.names=FALSE,quote=FALSE)
