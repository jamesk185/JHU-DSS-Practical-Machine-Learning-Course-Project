---
title: "Practical Macine Learning Course Project"
author: "James Kowalik"
date: "16/08/2021"
output: html_document
---

## Project Outline

This project was completed as part of the Practical Machine Learning course run by Johns Hopkins University. The following summary is as it was detailed in the project outline. 

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, my goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 

## Preprocessing

First, I will load the `caret` package and then download the data from the source and store it in respective training and testing objects.

```{r, message=FALSE, warning=FALSE}
library(caret)

trainingURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testingURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

download.file(trainingURL, "./pml-training.csv")
download.file(testingURL, "./pml-testing.csv")

traindata <- read.csv("./pml-training.csv")
testdata <- read.csv("./pml-testing.csv")
```

Next I will examine the structure of the data and the dimensions.

```{r}
str(traindata)
dim(traindata)
dim(testdata)
```

There are a number of variables that appear to be falsely factorised and a number that seem to be dominated by `NA`s.

```{r}
colSums(is.na(traindata))
```

This closer look shows that variables with `NA` values have precisely the same number of `NA` and thus it can be concluded that these variables are unnecessary.

```{r}
noNA <- !(colSums(is.na(traindata)) > 0)
traindata <- traindata[noNA]
testdata <- testdata[noNA]
dim(traindata)
dim(testdata)
```

Next I will remove the first 7 variables from the data frames as these are not the measurings we are interested in.

```{r}
traindata <- traindata[,-(1:7)]
testdata <- testdata[,-(1:7)]
dim(traindata)
dim(testdata)
```

Any variables with little variability will not be good predictors so, using the appropriate `caret` function, I will remove any variables that have near zero variance.

```{r}
nzv <- nearZeroVar(traindata, saveMetrics = TRUE)
traindata <- traindata[,nzv$nzv==FALSE]
testdata <- testdata[,nzv$nzv==FALSE]
dim(traindata)
dim(testdata)
```

## Model Fitting

In this section I will explore how 3 models fare. I have gone for three models that are generally considered to be the most accurate in machine learning; prediction trees, random forests and boosting.

One side note is that two additional methods were explored and have not been included in this project report. Principal Components Analysis seemed appropriate as there is a fairly large number of variables, however it proved not to improve the accuracy of the models and is very costly in the time it takes to build a model. Model stacking also didn't improve model accuracy and is unnecessary as the models originally already perform well.

I will create a partition in the dataset to facilitate use of a training and a testing set.

```{r}
set.seed(814)
inTrain <- createDataPartition(traindata$classe, p=0.7, list = FALSE)
training <- traindata[inTrain,]
testing <- traindata[-inTrain,]
dim(training)
dim(testing)
```

I will set a train control variable to allow for k-fold cross validation. This will provide a better estimate of the prediction errors. I will opt for a 3-fold cross validation.

```{r}
set.seed(815)
control <- trainControl(method="cv", number=3, verboseIter=F)
```

First I will try a prediction trees model and print the out of sample error.

```{r}
TREEfit <- train(classe~., method="rpart", data=training, tuneLength = 50, trControl=control)
TREEpred <- predict(TREEfit, testing)
1-(sum(TREEpred==testing$classe)/length(testing$classe))
```

Next I will try random forests model.

```{r}
RFfit <- train(classe~., method="rf", data=training, trControl=control)
RFpred <- predict(RFfit, testing)
1-(sum(RFpred==testing$classe)/length(testing$classe))
```

Next I will try a boosting model.

```{r}
GBMfit <- train(classe~., method="gbm", data=training, trControl=control, verbose=FALSE)
GBMpred <- predict(GBMfit, testing)
1-(sum(GBMpred==testing$classe)/length(testing$classe))
```

The random forests model provides the smallest out of sample error so I will select and proceed with that model.

## Result

Finally I will apply the model to the provided test data. 

```{r}
FINALpred <- predict(RFfit, testdata)
FINALpred
```
