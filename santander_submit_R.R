#########################################################################################################
# Load libraries
#########################################################################################################

# set working directory
setwd("C:/Projects/Santander")

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

#########################################################################################################
# Exploratory data analysis
#########################################################################################################

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

# density plot function
plotDensity <- function(variableNames) {
  a <- ggplot(santanderTrain, aes(x = variableNames))
  a  + geom_histogram(binwidth = .5, colour="black", fill="white")
  a + geom_density(alpha = .2, fill = "#FF6666") +
    geom_vline(aes(xintercept = mean(variableNames)),
      linetype = "dashed", size = 0.6
    )
}

# histogram for few variables 
plotDensity(santanderTrain$var_0)
plotDensity(santanderTrain$var_1)
plotDensity(santanderTrain$var_2)
plotDensity(santanderTrain$var_3)

# convert target variable to factor
santanderTrain$target <- as.factor(santanderTrain[, 201])

#########################################################################################################
# missing value analysis
#########################################################################################################

# Find missing value
sum(is.na(santanderTrain))

# sum of missing values are zero
# so no need to do further process for missing value analysis

#########################################################################################################
# outlier analysis
#########################################################################################################

# find numeric index from all the veriables
numericIndex <- sapply(santanderTrain, is.numeric)

# fetch numeric data from data frame
numericData <- santanderTrain[, numericIndex]

# store numeric column names
numericColNames <- colnames(numericData)

# draw box plot
for (i in 1:length(numericColNames)) {
  assign(paste0("san", i), ggplot(aes_string(y = (numericColNames[i]), x = "target"), data = subset(santanderTrain)) + stat_boxplot(geom = "errorbar", width = 0.5) +
    geom_boxplot(outlier.colour = "red", fill = "gray", outlier.shape = 18, outlier.size = 1, notch = FALSE) +
    theme(legend.position = "bottom") +
    labs(y = numericColNames[i], x = "target") +
    ggtitle(paste("Box plot of target for", numericColNames[i])))
}
  

## plotting plots together for few variables
gridExtra::grid.arrange(san1, san2, san3, san4, san5, san6, san7, san8, san9, san10, ncol = 10)
gridExtra::grid.arrange(san11, san12, ncol = 2)

# Handle outlier in two ways 
# 1) Remove outliers 
# 2) Assign NA and impute outliers
# If we remove outlier from this dataset - then this dataset will be imbalance
# So, we will replace outliers with NA
for (i in numericColNames) {
  temp_val <- santanderTrain[, i][santanderTrain[, i] %in% boxplot.stats(santanderTrain[, i])$out]
  santanderTrain[, i][santanderTrain[, i] %in% temp_val] <- NA
  print(i)
  print(paste("count# ", (length(temp_val))))
}

# view missing values
# create dataframe with missing parcentage
missingVal = data.frame(apply(santanderTrain, 2,function(x){sum(is.na(x))}))

#convert row names into col
missingVal$Columns = row.names(missingVal)
row.names(missingVal) = NULL

#assign col name
names(missingVal)[1] = "missing_percentage"

#Calculate percentage
missingVal$missing_percentage = (missingVal$missing_percentage/nrow(santanderTrain)) * 100

# Arrange in descending order
missingVal = missingVal[order(-missingVal$missing_percentage),]

# Rearrange columns
missingVal = missingVal[,c(2,1)]

# bar plot for view missing values
ggplot(missingVal, aes(x = missingVal$Columns, y = missingVal$missing_percentage)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  theme_gray() +
  xlab("features") + ylab("percentage") +
  scale_y_continuous(breaks = pretty_breaks(n = 10)) +
  ggtitle("Santander data analysis") +
  theme(text = element_text(size = 15))

# missing values imputed by mean method
# Mean method for imputation
for (i in numericColNames) {
  print(i)
  santanderTrain[, i][is.na(santanderTrain[, i])] <- mean(santanderTrain[, i], na.rm = T)
}


#########################################################################################################
# Feature selection
#########################################################################################################

# correlation plot
corrgram(santanderTrain[,numericColNames], order = F, upper.panel = panel.pie, text.panel = panel.txt, main="Correlation plot")

#########################################################################################################
# Handle imbalance dataset
#########################################################################################################

# View a Barplot for check counts of both the class 
ggplot(santanderTrain, aes_string(x = santanderTrain$target)) +
  geom_bar(stat = "count", fill = "steelblue") +
  theme_gray() +
  xlab("target") + ylab("count") +
  scale_y_continuous(breaks = pretty_breaks(n = 10)) +
  ggtitle("Santander data analysis") +
  theme(text = element_text(size = 15))

# count of target class
table(santanderTrain$target)

# probabilty of target class
prop.table(table(santanderTrain$target))

# Our dataset is biased to class zero
# Need to handle this biased problem
# Use SMOTE technique to balance target class
santanderBalanced <- ROSE(target ~ ., data = santanderTrain, seed = 1)$data

# Now we have balanced dataset
# count of target class
table(santanderBalanced$target)

# probabilty of target class
prop.table(table(santanderBalanced$target))

#########################################################################################################
# Feature scaling
#########################################################################################################

# Normality check for few variables
qqnorm(santanderBalanced$var_0)
qqnorm(santanderBalanced$var_1)
qqnorm(santanderBalanced$var_2)

# Normalisation of dataset
for (i in numericColNames) {
  print(i)
  santanderBalanced[, i] <- (santanderBalanced[, i] - min(santanderBalanced[, i])) / (max(santanderBalanced[, i] - min(santanderBalanced[, i])))
}

#########################################################################################################
# Logistic Regression
#########################################################################################################

# prepare data to build a model
# Random sampling
santanderSample <- santanderBalanced[sample(nrow(santanderBalanced), 50000, replace = F), ]

# split dataset into two parts
# Train - for build model
# Test - for validate model
trainIndex <- createDataPartition(santanderSample$target, p = .8, list = FALSE)
train <- santanderSample[trainIndex, ]
test <- santanderSample[-trainIndex, ]

# build logistic model with binomial family
logisticRegresssionModel <- glm(target ~ ., data = train, family = "binomial")

# view summary of logistic regression model
summary(logisticRegresssionModel)

# pridict test data using logistic regression
logitPrediction <- predict(logisticRegresssionModel, newdata = test, type = "response")

# convert prob to class
logitPrediction <- ifelse(logitPrediction > 0.5, 1, 0)

# build confusion matrix
confusionMatrixLogit <- table(Actual = test$target, Predicted = logitPrediction)
confusionMatrix(confusionMatrixLogit)

# accuracy
logitAccuracy <- sum(diag(confusionMatrixLogit)) / sum(confusionMatrixLogit)
logitAccuracy

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
# performance evaluation
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

#########################################################################################################
# Random Forest
#########################################################################################################

# build random forest model
randomForestModel <- randomForest(target ~ ., data = train, importance = TRUE, ntree = 150, mtry = 14)

# view model details
print(randomForestModel)

# view attribute
attributes(randomForestModel)

# View tree structure
getTree(randomForestModel,1, labelVar = TRUE)

# plot variable importance graph
varImpPlot(randomForestModel, pch = 5, col = "red", sort = TRUE, n.var = 5, main = "variable importance")

# view variable importance table
importance(randomForestModel)

# View random forest model summary
summary(randomForestModel)

# predict test data using Random forest model
RandomForestPredication <- predict(randomForestModel, test[-201])

# build confusion matrix
confusionMatrixRF <- table(Actual = test$target, predicted = RandomForestPredication)
confusionMatrix(confusionMatrixRF)

# accuracy
accuracyRF <- sum(diag(confusionMatrixRF)) / sum(confusionMatrixRF)

# misclassifiaction
missClassificationRF <- 1 - accuracyRF

# view error rate vs no of trees
plot(randomForestModel)

# find best optimal mtry from this plot
t <- tuneRF(train[, -201], train[, 201], stepFactor = 0.5, plot = TRUE, ntreeTry = 150, trace = TRUE, improve = 0.05)

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

# --------------------
# performance evaluation
# --------------------
# Tree# 150 + Mtry = 14
# accuracy = 71.27
# FNR = 28.22
# misclassification - 30.94
# recall - 69.71
# precision - 66.94
# OOB error - 32.33
# AUC - 0.7102

#########################################################################################################
# Naive Bayes
#########################################################################################################

# Naive bayes model development
naiveBayesModel = naiveBayes(target ~. , data = train)

# view model 
naiveBayesModel

# predict target for test dataset
# type - class or raw for probability
naiveBayesPredict = predict(naiveBayesModel, test[,1:200], type="class")

# build confusion matrics
confusionMatrixNB = table(predicted = naiveBayesPredict,Actual = test[,201])
confusionMatrix(confusionMatrixNB)

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

# ---------------------
# performance evaluation
# ---------------------
# accuracy = 77.09
# FNR = 22.91
# misclassification - 24.10
# recall - 75.46
# precision - 77.94
# AUC - 0.7102
# --------------------
####################################################################################################################################################
