# remove all the objects
rm(list = ls())
# remove except
rmExcept("santanderTrain")

# set working directory
setwd("C:/Users/chetan_hirapara/Downloads/Chicks/Learning/Edwisor_main/Projects")
# verify working directory
getwd()

# load libraries
libraryList <- c(
  "ggplot2",
  "corrgram",
  "DMwR",
  "caret",
  "randomForest",
  "unbalanced",
  "C50",
  "MASS",
  "rpart",
  "gbm",
  "ROSE",
  "e1071",
  "Information",
  "sampling",
  "scales",
  "psych",
  "ROCR",
  "pROC"
)

lapply(libraryList, require, character.only = TRUE)

## load data
santanderTrain <- read.csv("train.csv", header = T, na.strings = c(" ", "", "NA", NA))
santanderTest <- read.csv("test.csv", header = T, na.strings = c(" ", "", "NA", NA))

# view data structure
str(santanderTrain)

# view dimention of data
dim(santanderTrain)

# remove identifier variable(ID_code variable) as it can't help in prediction
santanderTrain <- subset(santanderTrain, select = -c(ID_code))

# re arrange columns into dataframe
santanderTrain <- santanderTrain[, c(2:201, 1)]
# view column name order
print(colnames(santanderTrain))

# Data manipulation : convert string categories into numeric
for (i in 1:ncol(santanderTrain)) {
  if (class(santanderTrain[, i]) == "factor") {
    print(santanderTrain[, i])
    santanderTrain[, i] <- factor(santanderTrain[, i], labels = (1:length(levels(factor(santanderTrain[, i])))))
  }
}

# convert target variable to factor
santanderTrain$target <- as.factor(santanderTrain[, 201])


# take backup
backupSantanderTrain <- santanderTrain

#########################################################################################################
### missing value analysis ###
# Find missing value
sum(is.na(santanderTrain))

# sum of missing values are zero
# so no need to do further missing value analysis
#########################################################################################################
### outlier analysis ###
numericIndex <- sapply(santanderTrain, is.numeric)
numericData <- santanderTrain[, numericIndex]
numericColNames <- colnames(numericData)

# draw box plot
# for (i in 1:length(numericColNames)) {
#   assign(paste0("san", i), ggplot(aes_string(y = (numericColNames[i]), x = "target"), data = subset(data_sample)) + stat_boxplot(geom = "errorbar", width = 0.5) +
#     geom_boxplot(outlier.colour = "red", fill = "gray", outlier.shape = 18, outlier.size = 1, notch = FALSE) +
#     theme(legend.position = "bottom") +
#     labs(y = numericColNames[i], x = "target") +
#     ggtitle(paste("Box plot of target for", numericColNames[i])))
# }
#
#
# ## plotting plots together
# gridExtra::grid.arrange(san1, san2, san3, san4, san5, san6, san7, san8, san9, san10, ncol = 10)
# gridExtra::grid.arrange(san11, san12, ncol = 2)

# Handle outlier 1) Remove 2) assign NA and impute
# ** If we remove outlier from this dataset - then this dataset will be imbalance
# remove outlier from all the variables
# cnt <- 0
# for (i in numericColNames) {
#   print(i)
#   outValues <- santanderTrain[, i][santanderTrain[, i] %in% boxplot.stats(santanderTrain[, i])$out]
#   print(length(outValues))
#   cnt <- cnt + length(outValues)
#   santanderTrain <- santanderTrain[which(!santanderTrain[, i] %in% outValues), ]
# }

# OR replace outliers with NA
for (i in numericColNames) {
  temp_val <- santanderTrain[, i][santanderTrain[, i] %in% boxplot.stats(santanderTrain[, i])$out]
  santanderTrain[, i][santanderTrain[, i] %in% temp_val] <- NA
}

# view missing values
missing_row <- santanderTrain[!complete.cases(santanderTrain), ]
head(missing_row)

### add values to missing places
# use knn imputation
# santanderTrain <- knnImputation(santanderTrain[, numericColNames], k = 7)
# this will not work due we have large amount of missing values
#   --- Either add some more observation or use Central tendancy methods for impute

# Mean method for imputation
for (i in numericColNames) {
  print(i)
  santanderTrain[, i][is.na(santanderTrain[, i])] <- mean(santanderTrain[, i], na.rm = T)
}

# Median method
# for (i in numericColNames) {
#   print(i)
#   santanderTrain[, i][is.na(santanderTrain[, i])] <- median(santanderTrain[, i], na.rm = T)
# }

# missing value impuatation via different methods ####################
# santanderTrain[4, 5] <- NA
# Actual value
# santanderTrain[4,2] = 11.0604
# KNN santanderTrain[4,2]
# Mean santanderTrain[4,2] = 10.6794
# Median santanderTrain[4,2] = 10.5245

# back up post outlier
backupPostOutlier <- santanderTrain

#########################################################################################################
### Handle imbalance dataset
# Bar plot(categorical data)
# if you want count then stat="count"
ggplot(santanderTrain, aes_string(x = santanderTrain$target)) +
  geom_bar(stat = "count", fill = "steelblue") +
  theme_gray() +
  xlab("target") + ylab("count") +
  scale_y_continuous(breaks = pretty_breaks(n = 10)) +
  ggtitle("Santander data analysis") +
  theme(text = element_text(size = 15))

table(santanderTrain$target)

# find probabilty of class
prop.table(table(santanderTrain$target))

# SMOTE
santanderBalanced <- ROSE(target ~ ., data = santanderTrain, seed = 1)$data
table(santanderBalanced$target)
prop.table(table(santanderBalanced$target))

# Now we have balanced dataset
#########################################################################################################
### Feature selection
## correlation plot
# corrgram(santanderTrain[,numericColNames], order = F, upper.panel = panel.pie, text.panel = panel.txt, main="Correlation plot")


####################################################################################################################################################
### Feature scaling

# Normality check
qqnorm(santanderBalanced$var_0)
qqnorm(santanderBalanced$var_1)
hist(santanderBalanced$var_3)

# Normalisation
for (i in numericColNames) {
  print(i)
  santanderBalanced[, i] <- (santanderBalanced[, i] - min(santanderBalanced[, i])) / (max(santanderBalanced[, i] - min(santanderBalanced[, i])))
}

####################################################################################################################################################
### Logistic Regression

# prepare data for model
# Sample random sampling
santanderSample <- santanderBalanced[sample(nrow(santanderBalanced), 50000, replace = F), ]

# createDataPartition - for stratified sampling
# param1 - reference variable
# param2 - test and train %
# param3 - include repetative rows
trainIndex <- createDataPartition(santanderSample$target, p = .8, list = FALSE)
train <- santanderSample[trainIndex, ]
test <- santanderSample[-trainIndex, ]

logisticRegresssionModel <- glm(target ~ ., data = train, family = "binomial")
summary(logisticRegresssionModel)

# pridict using logistic regression
logitPrediction <- predict(logisticRegresssionModel, newdata = test, type = "response")

# convert prob to class
logitPrediction <- ifelse(logitPrediction > 0.5, 1, 0)

# confusion matrix
confusionMatrixLogit <- table(Actual = test$target, Predicted = logitPrediction)
confusionMatrix(confusionMatrixLogit)

# accuracy
logitAccuracy <- sum(diag(confusionMatrixLogit)) / sum(confusionMatrixLogit)
# misclassification
missClassification <- 1 - logitAccuracy

# ROC
logitPrediction <- prediction(logitPrediction, test$target)
logitRoc <- performance(logitPrediction, "tpr", "fpr")
plot(logitRoc,
  colorize = T,
  main = "ROC Curve"
)
abline(a = 0, b = 1)

# AUC
logitAuc <- performance(logitPrediction, "auc")
logitAuc <- unlist(slot(logitAuc, "y.values"))
logitAuc <- round(logitAuc, 4)
# for print AUC on plot
legend(.3, .35, logitAuc, title = "AUC", cex = 1.2)

# ---------------
# Logistic regression with balanced data + Random sampling
# accuracy - 71.63
# FNR -  28.84
# missClassification - 28.37
# precision - 72.88
# recall - 71.16
# AIC - 44645
# AUC - 0.7163
# --------------

####################################################################################################################################################
### Random Forest ###
#
# santanderSample <- subset(santanderSample, select = -c(
#   var_142, var_34, var_106, var_198, var_196, var_159, var_61, var_74, var_75, var_7, var_53,
#   var_131, var_79, var_150, var_179, var_93, var_165, var_40, var_198, var_137, var_99,
#   var_128, var_60, var_44, var_67, var_78, var_73, var_65, var_6, var_12, var_1, var_188,
#   var_185, var_192, var_133, var_76, var_108, var_22, var_123, var_80, var_2, var_166, var_169
# ))

# build random forest model
randomForestModel <- randomForest(target ~ ., data = train, importance = TRUE, ntree = 150, mtry = 14)

# view model details
print(randomForestModel)
# view attribute
attributes(randomForestModel)

# View tree structure
getTree(randomForestModel,1, labelVar = TRUE)

varImpPlot(randomForestModel, pch = 5, col = "red", sort = TRUE, n.var = 30, main = "variable importance")

# importance(randomForestModel)
# varUsed(randomForestModel)
# summary(randomForestModel)

# predict test data using RF model
RandomForestPredication <- predict(randomForestModel, test[-201])

# evaluate perofmace of classification model
confusionMatrixRF <- table(Actual = test$target, predicted = RandomForestPredication)
confusionMatrix(confusionMatrixRF)

# accuracy
accuracyRF <- sum(diag(confusionMatrixRF)) / sum(confusionMatrixRF)
# Misclassifiaction
missClassificationRF <- 1 - accuracyRF

# view plot
plot(randomForestModel)

# Tune mtry
# find best optimal mtry from this plot
t <- tuneRF(train[, -201], train[, 201], stepFactor = 0.5, plot = TRUE, ntreeTry = 200, trace = TRUE, improve = 0.05)

# no of nodes
hist(treesize(randomForestModel), main = "No of nodes for the trees", col = "steelblue")

# ROC
RandomForestPredication <- prediction(as.numeric(RandomForestPredication), as.numeric(test$target))
RFRoc <- performance(RandomForestPredication, "tpr", "fpr")
plot(RFRoc,
     colorize = T,
     main = "ROC Curve"
)
abline(a = 0, b = 1)

# AUC
RFAuc <- performance(RandomForestPredication, "auc")
RFAuc <- unlist(slot(RFAuc, "y.values"))
RFAuc <- round(RFAuc, 4)
# for print AUC on plot
legend(.3, .35, RFAuc, title = "AUC", cex = 1.2)

# -------------
# Tree# 100 + Mtry = 14
# accuracy - 69.70
# FNR - 30.23
# misclassification - 30.30
# recall - 69.77
# precision - 69.69
# --------------------
# Tree# 150 + Mtry = 14
# accuracy = 71.27
# FNR = 28.22
# misclassification - 30.94
# recall - 69.71
# precision - 66.94
# OOB error - 32.33
# AUC - 0.7102

####################################################################################################################################################
### Naive Bayes ###

# Model development
naiveBayesModel = naiveBayes(target ~. , data = train)

# view model 
naiveBayesModel

# prediction
# type - class or raw for probability
naiveBayesPredict = predict(naiveBayesModel, test[,1:200], type="class")

# confusion matrics
confusionMatrixNB = table(predicted = naiveBayesPredict,Actual = test[,201])
# view statistics
confusionMatrix(confusionMatrixNB, positive = '1')

# ROC
naiveBayesPredict = predict(naiveBayesModel, test[,1:200])
naiveBayesPredict <- prediction(as.numeric(naiveBayesPredict), as.numeric(test$target))
nbRoc <- performance(naiveBayesPredict, "tpr", "fpr")
plot(nbRoc,
     colorize = T,
     main = "ROC Curve"
)
abline(a = 0, b = 1)

# AUC
nbAuc <- performance(naiveBayesPredict, "auc")
nbAuc <- unlist(slot(nbAuc, "y.values"))
nbAuc <- round(nbAuc, 4)
# for print AUC on plot
legend(.3, .35, nbAuc, title = "AUC", cex = 1.2)

# accuracy = 77.09
# FNR = 22.91
# misclassification - 24.10
# recall - 75.46
# precision - 77.94
# AUC - 0.7102


####################################################################################################################################################
### KNN Model
####################################################################################################################################################

# KNN prediction
KNNPredictionProb = knn(train[,1:200], test[,1:200], train$target, k = 3, prob = TRUE)

# Conf matrix
confusionMatrixKNN = table(predicted = KNNPredictionProb,actual = test$target)

confusionMatrix(confusionMatrixKNN)

# accuracy
sum(diag(confusionMatrixKNN))/nrow(test)

# plot AUC
library(pROC)

roc(as.numeric(test$target), as.numeric(KNNPredictionProb))

plot(roc(as.numeric(test$target), as.numeric(KNNPredictionProb)),
     print.thres = T,
     print.auc=T)

##### For k= 3 ######
# accuracy - 69.05
# FNR - 8.57
# recall - 91.43
# precision - 63.25
# AUC - 0.699

##### For k= 5 ######
# accuracy - 63.09
# FNR - 5.02
# recall - 94.98
# precision - 58.07
# AUC -

##### For k= 7 ######
# accuracy - 59.18
# FNR - 3.13
# recall - 96.87
# precision - 55.31
# AUC -

####################################################################################################################################################
