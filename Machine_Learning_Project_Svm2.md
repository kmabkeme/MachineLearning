# Machine Learning to Monitor Exercise Quality


Kristin Abkemeier, January 2018
===============================

## Synopsis

This assignment entails analyzing data from accelerometers placed on the belt, forearm, upper arm, and dumbbell of 6 people who performed dumbbell lifts correctly and then incorrectly in five different specific ways. The dumbbell lift motions were categorized into classes A, B, C, D, or E based on how the subject performed the exercise. I sought to find a machine learning algorithm that could be trained on the properly cleaned data so that the algorithm could correctly identify how the dumbbell lift was done according to class (called "classe" in this data set). After paring down the given data set to 53 usable columns and exploring several different machine learning approaches, I found that I could achieve accuracy of over 99 percent from using a support vector machine model that was tuned with the correct model type and parameters. 

## Exploring and Cleaning the Data

For this report, I read in the data contained in the file pml-training.csv that was collected in the research by Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; and Fuks, H., "Qualitative Activity Recognition of Weight Lifting Exercises, 2013 [1]. 

I want to note that I am trying to write as original a report as possible to do while fulfilling the requirements of the assignment. I used the R language (https://www.r-project.org/) and RStudio (https://www.rstudio.com/) to do the computations for this assignment. To solve the problem, I used the functions included in the base packages that come included with R and the e1071 package, called Misc Functions of the Department of Statistics, Probability Theory Group (Formerly E1071)[2]. It feels strange to use such a powerful canned function as svm() to do an assignment, but I doubt that I am expected to write my own machine learning algorithm for this course, so I use the tools that others have created before me, and I am giving the creators credit.

I began my analysis by reading the pml-training.csv data set into RStudio and then loading in the e1071 package.

```r
## Loading and preprocessing the data
setwd("/Users/kristinabkemeier/DataScience/Practical Machine Learning/Project")
actData <- read.csv("pml-training.csv", header=TRUE)
## And read in the library that I will need to use, from the e1071 package by Meyer, D.; Dimitriadou, E.; Hornik, K.; Weingessel, A.; Leisch, F.; Chang, C.-C.; Lin, C.-C. [e1071: Misc Functions of the Department of Statistics, Probability Theory Group (Formerly E1071), TU Wien](https://CRAN.R-project.org/package=e1071) Version 1.6-8, published 2017-02-02 at https://CRAN.R-project.org/package=e1071 
library(e1071)
```

Because the table started with 160 columns of data, I did an initial visual inspection of the table in Excel to see if I could identify any patterns that would allow me to get rid of the columns that would not contribute towards a good fit of the data. Some columns were sparsely populated, with only occasional nonzero entries that contained quantities that contained values calculated from multiple rows. However, a look at the testing data set provided, pml-testing.csv, indicated that we would only need to classify entries according to the values in individual rows, with no cross-correlations indicated. So, it was clear that the sparsely populated columns were likely unnecessary.

I identified all of the columns that had NAs in them. It turned out that all of the columns with NA values in the first row were almost entirely filled with NA values. I checked the percentage of values in the column that were "NA".

```r
hasNA <- which(is.na(actData[1,]))
length(hasNA)
```

```
## [1] 67
```

```r
hasNA
```

```
##  [1]  18  19  21  22  24  25  27  28  29  30  31  32  33  34  35  36  50
## [18]  51  52  53  54  55  56  57  58  59  75  76  77  78  79  80  81  82
## [35]  83  93  94  96  97  99 100 103 104 105 106 107 108 109 110 111 112
## [52] 131 132 134 135 137 138 141 142 143 144 145 146 147 148 149 150
```

```r
for (i in rev(hasNA))
{
  percentIsNA <- sum(is.na(actData[,i]))/length(actData[,i]) * 100
  print(i)
  print(percentIsNA)
}
```

```
## [1] 150
## [1] 97.93089
## [1] 149
## [1] 97.93089
## [1] 148
## [1] 97.93089
## [1] 147
## [1] 97.93089
## [1] 146
## [1] 97.93089
## [1] 145
## [1] 97.93089
## [1] 144
## [1] 97.93089
## [1] 143
## [1] 97.93089
## [1] 142
## [1] 97.93089
## [1] 141
## [1] 97.93089
## [1] 138
## [1] 97.93089
## [1] 137
## [1] 97.93089
## [1] 135
## [1] 97.93089
## [1] 134
## [1] 97.93089
## [1] 132
## [1] 97.93089
## [1] 131
## [1] 97.93089
## [1] 112
## [1] 97.93089
## [1] 111
## [1] 97.93089
## [1] 110
## [1] 97.93089
## [1] 109
## [1] 97.93089
## [1] 108
## [1] 97.93089
## [1] 107
## [1] 97.93089
## [1] 106
## [1] 97.93089
## [1] 105
## [1] 97.93089
## [1] 104
## [1] 97.93089
## [1] 103
## [1] 97.93089
## [1] 100
## [1] 97.93089
## [1] 99
## [1] 97.93089
## [1] 97
## [1] 97.93089
## [1] 96
## [1] 97.93089
## [1] 94
## [1] 97.93089
## [1] 93
## [1] 97.93089
## [1] 83
## [1] 97.93089
## [1] 82
## [1] 97.93089
## [1] 81
## [1] 97.93089
## [1] 80
## [1] 97.93089
## [1] 79
## [1] 97.93089
## [1] 78
## [1] 97.93089
## [1] 77
## [1] 97.93089
## [1] 76
## [1] 97.93089
## [1] 75
## [1] 97.93089
## [1] 59
## [1] 97.93089
## [1] 58
## [1] 97.93089
## [1] 57
## [1] 97.93089
## [1] 56
## [1] 97.93089
## [1] 55
## [1] 97.93089
## [1] 54
## [1] 97.93089
## [1] 53
## [1] 97.93089
## [1] 52
## [1] 97.93089
## [1] 51
## [1] 97.93089
## [1] 50
## [1] 97.93089
## [1] 36
## [1] 97.93089
## [1] 35
## [1] 97.93089
## [1] 34
## [1] 97.93089
## [1] 33
## [1] 97.93089
## [1] 32
## [1] 97.93089
## [1] 31
## [1] 97.93089
## [1] 30
## [1] 97.93089
## [1] 29
## [1] 97.93089
## [1] 28
## [1] 97.93089
## [1] 27
## [1] 97.93089
## [1] 25
## [1] 97.93089
## [1] 24
## [1] 97.93089
## [1] 22
## [1] 97.93089
## [1] 21
## [1] 97.93089
## [1] 19
## [1] 97.93089
## [1] 18
## [1] 97.93089
```

Likewise, many columns were filled with mostly blanks, so I also screened for those and calculated the percentages of empty values.

```r
isEmpty <- which((actData[1,]==""))
length(isEmpty)
```

```
## [1] 33
```

```r
isEmpty
```

```
##  [1]  12  13  14  15  16  17  20  23  26  69  70  71  72  73  74  87  88
## [18]  89  90  91  92  95  98 101 125 126 127 128 129 130 133 136 139
```

```r
for (i in rev(isEmpty))
{
  percentIsEmpty <- sum(actData[,i]=="")/length(actData[,i]) * 100
  print(i)
  print(percentIsEmpty)
}
```

```
## [1] 139
## [1] 97.93089
## [1] 136
## [1] 97.93089
## [1] 133
## [1] 97.93089
## [1] 130
## [1] 97.93089
## [1] 129
## [1] 97.93089
## [1] 128
## [1] 97.93089
## [1] 127
## [1] 97.93089
## [1] 126
## [1] 97.93089
## [1] 125
## [1] 97.93089
## [1] 101
## [1] 97.93089
## [1] 98
## [1] 97.93089
## [1] 95
## [1] 97.93089
## [1] 92
## [1] 97.93089
## [1] 91
## [1] 97.93089
## [1] 90
## [1] 97.93089
## [1] 89
## [1] 97.93089
## [1] 88
## [1] 97.93089
## [1] 87
## [1] 97.93089
## [1] 74
## [1] 97.93089
## [1] 73
## [1] 97.93089
## [1] 72
## [1] 97.93089
## [1] 71
## [1] 97.93089
## [1] 70
## [1] 97.93089
## [1] 69
## [1] 97.93089
## [1] 26
## [1] 97.93089
## [1] 23
## [1] 97.93089
## [1] 20
## [1] 97.93089
## [1] 17
## [1] 97.93089
## [1] 16
## [1] 97.93089
## [1] 15
## [1] 97.93089
## [1] 14
## [1] 97.93089
## [1] 13
## [1] 97.93089
## [1] 12
## [1] 97.93089
```

Because imputation is only viable if there are values in a majority of rows (more than 50 percent), it made more sense to eliminate these rows comprising mostly NAs and empty values. Also, the first seven columns contained values that couldn't be used to include in a model, such as row index numbers, subject names, timestamps, and some additional non-quantitative criteria. Thus, I removed those columns as well.

```r
colsToRemove <- c(1:7, hasNA, isEmpty)
actDataSmaller <- actData[,-colsToRemove]
```

I needed to partition the training data set so that we could test our model before we tried testing it on the 20 testing values given in the file pml-testing.csv. I chose to put 75% of the pml-training.csv data into my training set and 25% into my testing set for checking my out of sample error rate. Because I am not using the Caret package in this project, I had to find a different way in which to partition the data instead of via the train() function. Instead, I just sent every fourth value to a validation data set. I planned to check the accuracy of my most successful model against the validation data set, called validationData here, before running the model on the actual testing data given with the assignment.

```r
numRows <- dim(actData)[1]
numRowsDividedBy4 <- numRows/4
indicesPart <- seq(numRowsDividedBy4)*4

validationData <- actDataSmaller[indicesPart,]
trainingData <- actDataSmaller[-indicesPart,]
```

## Machine Learning Attempts

If I knew how, I would write my own machine learning tool, but building unique and individual machine learning tools is not the aim of this course. So, I had to use the machine learning tools that I learned about in this course. Initially, I tried several different machine learning models in the Caret package. However several of these models ("bagEarth"", "bagFDA", random forest "rf" and generalized boosted regression model "gbm") did not converge after an hour. I am not including these results here, but suffice it to say that I found using the Caret package for this assignment to be problematic. Most of the functions I tried did not converge.

Instead, I decided to explore support vector machines in the e1071 package. A support vector machine (SVM) works by assigns new examples to categories. According to the Wikipedia article on SVMs,

>An SVM model is a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall.[3]

I needed to create the set of regressors for my training data set (trainingData) and validation data set (validationData).

```r
regressorsTrain <- subset(trainingData, select = -classe)
regressorsValidation <- subset(validationData, select = -classe)
```

According to the Wikipedia article, SVMs can efficiently perform a non-linear classification using what is called the kernel trick, implicitly mapping their inputs into high-dimensional feature spaces. The default kernel in svm() is linear, but I also tried the polynomial kernel, because I knew from exploratory plotting that the data from the six subjects was very complex.

First, I did the training of the SVM model by calling SVM on my regressors, using a linear kernel, and setting a k-fold cross-validation with k = 5 on the training data by setting the input parameter cross to the value of 5.

```r
## Set seed for reproducibility
set.seed(1342)

## Support vector matrix function is from the e1071 package by Meyer, D.; Dimitriadou, E.; Hornik, K.; Weingessel, A.; Leisch, F.; Chang, C.-C.; Lin, C.-C. [e1071: Misc Functions of the Department of Statistics, Probability Theory Group (Formerly E1071), TU Wien](https://CRAN.R-project.org/package=e1071) Version 1.6-8, published 2017-02-02 at https://CRAN.R-project.org/package=e1071 
## I just want to warn people that this is a course about practical machine learning, so we are not expected to write machine learning tools from scratch. Therefore, I am citing here that I am using the svm() function which the team of D. Meyer and others created as part of the e1071 package for R. I am making sure to give them all of the credit that they so richly deserve here for developing this powerful tool.

trainModel1 <- svm(regressorsTrain, trainingData$classe, kernel="linear", cross=5)
summary(trainModel1)
```

```
## 
## Call:
## svm.default(x = regressorsTrain, y = trainingData$classe, kernel = "linear", 
##     cross = 5)
## 
## 
## Parameters:
##    SVM-Type:  C-classification 
##  SVM-Kernel:  linear 
##        cost:  1 
##       gamma:  0.01923077 
## 
## Number of Support Vectors:  7630
## 
##  ( 1479 1764 1601 1373 1413 )
## 
## 
## Number of Classes:  5 
## 
## Levels: 
##  A B C D E
## 
## 5-fold cross-validation on training data:
## 
## Total Accuracy: 78.32439 
## Single Accuracies:
##  77.60788 78.59327 79.27989 79.00102 77.13995
```

Then, I did the predictions on the training set, using the predict() function from the R stats package included in the basic installation of R [4]. I looked at what is called the confusion matrix, which compares the predicted values in the trainingData$classe column with the actual values in the column. You can get the accuracy of the prediction by summing up the percentages in the diagonal entries of the matrix. I will call the matrix conf so that my sum over the diagonal takes up less space on the page.

```r
## The predict() function is included in the stats package included in the basic installation of R.
prediction1 <- predict(trainModel1, regressorsTrain)
conf <- table(prediction1, trainingData$classe)
accuracy1 <- (conf[1,1] + conf[2,2] + conf[3,3] + conf[4,4] + conf[5,5])/sum(conf[,])
accuracy1
```

```
## [1] 0.7915336
```

We see that the initial accuracy value for the SVM with a linear kernel is not very high. This did not seem surprising, given the complexity of the data, with the different motion patterns displayed by the six subjects for the five different ways of doing the exercise. So, I switched to a polynomial kernel, keeping the 5-fold cross-validation, and setting a couple of additional parameters that the function svm() allows when the kernel is type polynomial: degree = 3, so that a third-degree polynomial could be used (actually, this is the default value); and coef0 is the value in the polynomial kernel expression (gamma*u'*v + coef0)^degree given in the R help file for svm(). Setting coef0 to a nonzero value can apparently increase the boundaries between categories of points, so it is worth exploring the effect of changing it. I tried the values coef0 = 1, 2, 3, and then skipped to 20 because I saw a trend.

```r
trainModel2 <- svm(regressorsTrain, trainingData$classe, kernel="polynomial", cross=5, degree=3, coef0=1)
summary(trainModel2)
```

```
## 
## Call:
## svm.default(x = regressorsTrain, y = trainingData$classe, kernel = "polynomial", 
##     degree = 3, coef0 = 1, cross = 5)
## 
## 
## Parameters:
##    SVM-Type:  C-classification 
##  SVM-Kernel:  polynomial 
##        cost:  1 
##      degree:  3 
##       gamma:  0.01923077 
##      coef.0:  1 
## 
## Number of Support Vectors:  4448
## 
##  ( 827 1006 1101 820 694 )
## 
## 
## Number of Classes:  5 
## 
## Levels: 
##  A B C D E
## 
## 5-fold cross-validation on training data:
## 
## Total Accuracy: 96.82 
## Single Accuracies:
##  97.55352 96.90792 96.94293 95.85457 96.84103
```

```r
trainModel3 <- svm(regressorsTrain, trainingData$classe, kernel="polynomial", cross=5, degree=3, coef0=2)
summary(trainModel3)
```

```
## 
## Call:
## svm.default(x = regressorsTrain, y = trainingData$classe, kernel = "polynomial", 
##     degree = 3, coef0 = 2, cross = 5)
## 
## 
## Parameters:
##    SVM-Type:  C-classification 
##  SVM-Kernel:  polynomial 
##        cost:  1 
##      degree:  3 
##       gamma:  0.01923077 
##      coef.0:  2 
## 
## Number of Support Vectors:  3735
## 
##  ( 655 862 935 680 603 )
## 
## 
## Number of Classes:  5 
## 
## Levels: 
##  A B C D E
## 
## 5-fold cross-validation on training data:
## 
## Total Accuracy: 97.6218 
## Single Accuracies:
##  97.34964 97.45158 97.86005 97.85933 97.58832
```

```r
trainModel4 <- svm(regressorsTrain, trainingData$classe, kernel="polynomial", cross=5, degree=3, coef0=3)
summary(trainModel4)
```

```
## 
## Call:
## svm.default(x = regressorsTrain, y = trainingData$classe, kernel = "polynomial", 
##     degree = 3, coef0 = 3, cross = 5)
## 
## 
## Parameters:
##    SVM-Type:  C-classification 
##  SVM-Kernel:  polynomial 
##        cost:  1 
##      degree:  3 
##       gamma:  0.01923077 
##      coef.0:  3 
## 
## Number of Support Vectors:  3377
## 
##  ( 589 767 843 621 557 )
## 
## 
## Number of Classes:  5 
## 
## Levels: 
##  A B C D E
## 
## 5-fold cross-validation on training data:
## 
## Total Accuracy: 97.90039 
## Single Accuracies:
##  97.85933 97.96126 97.55435 97.96126 98.16576
```

```r
trainModel5 <- svm(regressorsTrain, trainingData$classe, kernel="polynomial", cross=5, degree=3, coef0=20)
summary(trainModel5)
```

```
## 
## Call:
## svm.default(x = regressorsTrain, y = trainingData$classe, kernel = "polynomial", 
##     degree = 3, coef0 = 20, cross = 5)
## 
## 
## Parameters:
##    SVM-Type:  C-classification 
##  SVM-Kernel:  polynomial 
##        cost:  1 
##      degree:  3 
##       gamma:  0.01923077 
##      coef.0:  20 
## 
## Number of Support Vectors:  2435
## 
##  ( 409 532 574 472 448 )
## 
## 
## Number of Classes:  5 
## 
## Levels: 
##  A B C D E
## 
## 5-fold cross-validation on training data:
## 
## Total Accuracy: 98.72257 
## Single Accuracies:
##  98.2331 99.18451 98.74321 98.60686 98.84511
```

The accuracy increases as coef0 increases. Then I checked what would happen if I increased the cross-validation from cross=5 to cross=10: 

```r
trainModel6 <- svm(regressorsTrain, trainingData$classe, kernel="polynomial", cross=10, degree=3, coef0=20)
summary(trainModel6)
```

```
## 
## Call:
## svm.default(x = regressorsTrain, y = trainingData$classe, kernel = "polynomial", 
##     degree = 3, coef0 = 20, cross = 10)
## 
## 
## Parameters:
##    SVM-Type:  C-classification 
##  SVM-Kernel:  polynomial 
##        cost:  1 
##      degree:  3 
##       gamma:  0.01923077 
##      coef.0:  20 
## 
## Number of Support Vectors:  2435
## 
##  ( 409 532 574 472 448 )
## 
## 
## Number of Classes:  5 
## 
## Levels: 
##  A B C D E
## 
## 10-fold cross-validation on training data:
## 
## Total Accuracy: 98.96039 
## Single Accuracies:
##  98.50442 99.04891 98.91304 99.11625 99.04891 99.11685 98.9123 98.84511 98.98098 99.11685
```

Changing the cross-validation number did not have an effect, but then changing the value of the parameter gamma from its default value of 1 to setting a value of 10 increased the accuracy further:

```r
trainModel7 <- svm(regressorsTrain, trainingData$classe, kernel="polynomial", cross=10, degree=3, coef0=20, gamma=10)
summary(trainModel7)
```

```
## 
## Call:
## svm.default(x = regressorsTrain, y = trainingData$classe, kernel = "polynomial", 
##     degree = 3, gamma = 10, coef0 = 20, cross = 10)
## 
## 
## Parameters:
##    SVM-Type:  C-classification 
##  SVM-Kernel:  polynomial 
##        cost:  1 
##      degree:  3 
##       gamma:  10 
##      coef.0:  20 
## 
## Number of Support Vectors:  3302
## 
##  ( 615 747 710 585 645 )
## 
## 
## Number of Classes:  5 
## 
## Levels: 
##  A B C D E
## 
## 10-fold cross-validation on training data:
## 
## Total Accuracy: 99.0759 
## Single Accuracies:
##  99.11625 98.77717 98.98098 99.04827 99.11685 99.04891 99.18423 99.25272 99.11685 99.11685
```

Finally, I noticed from ?svm that there was another parameter, cost, that is called the cost of constrains violation. It has a default value of 1, but I wanted to see what effect increasing the value of cost to 10 and then 100 might have:

```r
trainModel8 <- svm(regressorsTrain, trainingData$classe, kernel="polynomial", cross=10, degree=3, coef0=20, gamma=10, cost=10)
summary(trainModel8)
```

```
## 
## Call:
## svm.default(x = regressorsTrain, y = trainingData$classe, kernel = "polynomial", 
##     degree = 3, gamma = 10, coef0 = 20, cost = 10, cross = 10)
## 
## 
## Parameters:
##    SVM-Type:  C-classification 
##  SVM-Kernel:  polynomial 
##        cost:  10 
##      degree:  3 
##       gamma:  10 
##      coef.0:  20 
## 
## Number of Support Vectors:  3302
## 
##  ( 615 747 710 585 645 )
## 
## 
## Number of Classes:  5 
## 
## Levels: 
##  A B C D E
## 
## 10-fold cross-validation on training data:
## 
## Total Accuracy: 99.08269 
## Single Accuracies:
##  99.11625 99.04891 99.32065 98.64038 98.84511 99.32065 99.18423 99.11685 99.04891 99.18478
```

```r
trainModel9 <- svm(regressorsTrain, trainingData$classe, kernel="polynomial", cross=10, degree=3, coef0=20, gamma=10, cost=100)
summary(trainModel9)
```

```
## 
## Call:
## svm.default(x = regressorsTrain, y = trainingData$classe, kernel = "polynomial", 
##     degree = 3, gamma = 10, coef0 = 20, cost = 100, cross = 10)
## 
## 
## Parameters:
##    SVM-Type:  C-classification 
##  SVM-Kernel:  polynomial 
##        cost:  100 
##      degree:  3 
##       gamma:  10 
##      coef.0:  20 
## 
## Number of Support Vectors:  3302
## 
##  ( 615 747 710 585 645 )
## 
## 
## Number of Classes:  5 
## 
## Levels: 
##  A B C D E
## 
## 10-fold cross-validation on training data:
## 
## Total Accuracy: 99.06231 
## Single Accuracies:
##  99.04827 98.98098 98.91304 99.11625 99.04891 99.04891 99.18423 99.18478 99.04891 99.04891
```

We have gotten the accuracy to above 99 percent on the training in trainModel9, so let's compare how all of these models do for predicting the validation data set, validationData, before we try it on the actual test data set. I used the predict() function from the R stats package [4][stats] included in the basic installation of R. I looked at what is called the confusion matrix, which compares the predicted values in the trainingData$classe column with the actual values in the column. You can get the accuracy of the prediction by summing up the percentages in the diagonal entries of the matrix. I will call the matrix conf so that my sum over the diagonal takes up less space on the page.

```r
## predict() is a function in the stats package that comes with the base installation of R.[4][stats]
prediction1 <- predict(trainModel1, regressorsValidation)
prediction2 <- predict(trainModel2, regressorsValidation)
prediction3 <- predict(trainModel3, regressorsValidation)
prediction4 <- predict(trainModel4, regressorsValidation)
prediction5 <- predict(trainModel5, regressorsValidation)
prediction6 <- predict(trainModel6, regressorsValidation)
prediction7 <- predict(trainModel7, regressorsValidation)
prediction8 <- predict(trainModel8, regressorsValidation)
prediction9 <- predict(trainModel9, regressorsValidation)

conf <- table(prediction1, validationData$classe)
accuracy1 <- (conf[1,1] + conf[2,2] + conf[3,3] + conf[4,4] + conf[5,5])/sum(conf[,])
accuracy1
```

```
## [1] 0.7896024
```

```r
conf <- table(prediction2, validationData$classe)
accuracy2 <- (conf[1,1] + conf[2,2] + conf[3,3] + conf[4,4] + conf[5,5])/sum(conf[,])
accuracy2
```

```
## [1] 0.9761468
```

```r
conf <- table(prediction3, validationData$classe)
accuracy3 <- (conf[1,1] + conf[2,2] + conf[3,3] + conf[4,4] + conf[5,5])/sum(conf[,])
accuracy3
```

```
## [1] 0.9824669
```

```r
conf <- table(prediction4, validationData$classe)
accuracy4 <- (conf[1,1] + conf[2,2] + conf[3,3] + conf[4,4] + conf[5,5])/sum(conf[,])
accuracy4
```

```
## [1] 0.9847095
```

```r
conf <- table(prediction5, validationData$classe)
accuracy5 <- (conf[1,1] + conf[2,2] + conf[3,3] + conf[4,4] + conf[5,5])/sum(conf[,])
accuracy5
```

```
## [1] 0.9930683
```

```r
conf <- table(prediction6, validationData$classe)
accuracy6 <- (conf[1,1] + conf[2,2] + conf[3,3] + conf[4,4] + conf[5,5])/sum(conf[,])
accuracy6
```

```
## [1] 0.9930683
```

```r
conf <- table(prediction7, validationData$classe)
accuracy7 <- (conf[1,1] + conf[2,2] + conf[3,3] + conf[4,4] + conf[5,5])/sum(conf[,])
accuracy7
```

```
## [1] 0.9969419
```

```r
conf <- table(prediction8, validationData$classe)
accuracy8 <- (conf[1,1] + conf[2,2] + conf[3,3] + conf[4,4] + conf[5,5])/sum(conf[,])
accuracy8
```

```
## [1] 0.9969419
```

```r
conf <- table(prediction9, validationData$classe)
accuracy9 <- (conf[1,1] + conf[2,2] + conf[3,3] + conf[4,4] + conf[5,5])/sum(conf[,])
accuracy9
```

```
## [1] 0.9969419
```

The accuracies for the validation data sets increase as we change the various parameters, and the last three values are the same, which indicates that varying the cost parameter made no difference. Thus, trainModel7, with svm(regressorsTrain, trainingData$classe, kernel="polynomial", cross=10, degree=3, coef0=20, gamma=10), appears to be the best model we can achieve with an SVM without getting tremendously detailed. A 99% accuracy is good enough for getting the answers to the test data, which we read in and process in an identical fashion to how we handled the validation data set.


```r
actData <- read.csv("pml-testing.csv", header=TRUE)
actDataSmaller <- actData[,-colsToRemove]
regressorsTest <- subset(actDataSmaller, select = -problem_id)
predictionTest <- predict(trainModel7, regressorsTest)
predictionTest
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

I entered these values into the quiz and got 20 out of 20 correct. So, the support vector machine model worked.

## Conclusion

The support vector machine model svm() in the e1071 package of R can be used to identify the manner in which an exercise is performed with over 99 percent accuracy if it is properly modified with the parameters of degree, coef0, and gamma. The cost parameter turned out to be irrelevant in this case.

## References

[1]. Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; and Fuks, H., "Qualitative Activity Recognition of Weight Lifting Exercises, 2013. [Link retrieved on 2018-01-11. ](http://web.archive.org/web/20170519033209/http://groupware.les.inf.puc-rio.br:80/public/papers/2013.Velloso.QAR-WLE.pdf)

[2]. Meyer, D.; Dimitriadou, E.; Hornik, K.; Weingessel, A.; Leisch, F.; Chang, C.-C.; Lin, C.-C. [e1071: Misc Functions of the Department of Statistics, Probability Theory Group (Formerly E1071), TU Wien](https://CRAN.R-project.org/package=e1071) Version 1.6-8, published 2017-02-02 at https://CRAN.R-project.org/package=e1071.

[3]. https://en.wikipedia.org/wiki/Support_vector_machine, retrieved 2018-01-13.

[4]. The stats package is one of the add-on packages that comes with the base R distribution. Link: [https://cran.r-project.org/doc/FAQ/R-FAQ.html#Add_002don-packages-in-R](https://cran.r-project.org/doc/FAQ/R-FAQ.html#Add_002don-packages-in-R). 
