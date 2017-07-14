# Human Activity Recognition
José Cotrim  



## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## Getting and Cleaning Data

The data files for this project are available here:

- [Training Data](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)

- [Testing Data](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)

Downloading the data directly into memmory.


```r
train_URL <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv'
test_URL  <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv'

rawTraining <- read.csv(url(train_URL), na.strings=c("NA","#DIV/0!",""))
rawTesting  <- read.csv(url(test_URL), na.strings=c("NA","#DIV/0!",""))
```

The first 7 variables refers to identification of the observations like timestamp etc.

These variables will not be used as predictors so it will be removed.


```r
training <- rawTraining[,8:length(rawTraining)]
testing  <- rawTesting[ ,8:length(rawTesting )]
```

Identifying and removing variables where its contents are at least 90% of NAs.


```r
naVariables <- sapply(training, function(x) mean(is.na(x))) > .90       # Identifying NA variables
training <- training[, naVariables==FALSE]                              # Removing Variables
testing  <- testing[,  naVariables==FALSE]                              # Removing Variables
```

Check for near zero variation variables.


```r
nzv<- nearZeroVar(training,saveMetrics=TRUE)                            # Identifying NZV variables
training <- training[,nzv$nzv==FALSE]                                   # Removing Variables
testing  <- testing[ ,nzv$nzv==FALSE]                                   # Removing Variables
```

Summary of transformations:

- **7 first variables** that represents the identification of the observations were removed.

- **100 variable(s)** were removed from the datasets due to its contents are at least 90% with NA values.

- **0 variable(s)** was removed from the datasets due to its contents are near zero variation.

- It was left **53 variables** from  the initial **160 variables ** avaliable in the dataset.


## Prediction - Random Forest

As the random forest algorithm is an ensemble learning method for classification, it is appropriate for our problem.

The training data will be splitted in two datasets: 60% for training itself and 40% for cross validation testing.


```r
library(caret)
library(rpart)
library(randomForest)

# Training dataset partitioning
set.seed(1234)
inTrain <- createDataPartition(y=training$classe, p=0.6, list=FALSE)
myTraining <- training[ inTrain, ]
myTesting  <- training[-inTrain, ]

# Train on training set 1 of 4 with no extra features.
set.seed(1234)
mod1  <- train(classe ~ ., data = myTraining, method="rf", trControl=trainControl(method = "cv", number = 4))
pred1 <- predict(mod1, newdata=myTesting)
cm <- confusionMatrix(pred1, myTesting$classe)
cm
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2231   18    0    0    0
##          B    0 1494   11    1    2
##          C    1    6 1349   22    2
##          D    0    0    8 1261    4
##          E    0    0    0    2 1434
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9902          
##                  95% CI : (0.9877, 0.9922)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9876          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9996   0.9842   0.9861   0.9806   0.9945
## Specificity            0.9968   0.9978   0.9952   0.9982   0.9997
## Pos Pred Value         0.9920   0.9907   0.9775   0.9906   0.9986
## Neg Pred Value         0.9998   0.9962   0.9971   0.9962   0.9988
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1904   0.1719   0.1607   0.1828
## Detection Prevalence   0.2866   0.1922   0.1759   0.1622   0.1830
## Balanced Accuracy      0.9982   0.9910   0.9907   0.9894   0.9971
```

By the confusion matrix summary above, running the model on the test data for cross validation (myTraining) we get an **accuracy of 99.02%**.

The **expected out of sample error** is 100%-99.02% = **0.98%**.


## Predicting Test Cases

Using the prediction model in the testing dataset (20 different test cases) we get the following results:


```r
pred2 <- predict(mod1, testing)
pred2
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
