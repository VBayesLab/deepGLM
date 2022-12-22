# Examples demonstrate how to use deepGLM function to fit data with continuous dependent variable
#
# Copyright 2018
#                Nghia Nguyen (nghia.nguyen@sydney.edu.au)
#                Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au)
#
# https://github.com/VBayesLab/deepGLM
#
# Version: 1.0
# LAST UPDATE: May, 2018

# Clear all variables
rm(list=ls())
gc(reset=T)

# Load libs
library(mvtnorm)
library(rstudioapi)

RootDir <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(RootDir)

# Source external functions
source('dependencies.R')

# Read data file
data <- read.csv(file = "01_data/abalone.csv",header = FALSE)


# Divide data to training and test sets
N <- nrow(data)                               # Total number of observations
p <- ncol(data) - 1                           # Number of variables
Ntest <- round(0.15*N)                        # Number of test observations
idx <- sample.int(N, size = Ntest, replace = FALSE)    # Sampling indexes
dataTest <- data[idx,]                        # Test data
dataTrain <- data[-idx,]                      # Train data
XTrain <- data.matrix(dataTrain[,1:p])        # X train
y <- data.matrix(dataTrain[,p+1])                          # y train
XTest <- data.matrix(dataTest[,1:p])          # X test
yTest <- data.matrix(dataTest[,p+1])          # y test

# Normalize Train and Test data
meanX <- colMeans(XTrain)
stdX <- apply(XTrain, 2, sd)
X <- sweep(sweep(XTrain,2,meanX,'-'),2,stdX,'/')
XTest <- sweep(sweep(XTest,2,meanX,'-'),2,stdX,'/')

# Fit a deepGLM model
deepGLMout <-deepGLMfit(X,y,Network = c(5,5,5),Seed = 100,Verbose = 1, MaxEpoch = 500)

# Make prediction (point estimation) on a test set, without true labels
Pred1 <- deepGLMpredict(deepGLMout,XTest)

# If ytest is specified (for model evaluation purpose) then we can check PPS and MSE on test set
print('----------------Prediction---------------')
Pred2 <- deepGLMpredict(deepGLMout,XTest,y = yTest)
cat('PPS on test set using deepGLM is: ',Pred2$pps,'\n')
cat('MSE on test set using deepGLM is: ',Pred2$mse,'\n')

# You can also perform point and interval estimation for a single test observation
idx <- nrow(XTest)                                                     # Pick a random unseen observation
dataTest <- XTest[idx,]
Pred3 <- deepGLMpredict(deepGLMout,dataTest,Interval=1,Nsample=1000)    # Make 1-std prediction interval
cat('Prediction Interval: [',Pred3$interval[1],';',Pred3$interval[2],']','\n')
cat('True value: ',yTest[idx],'\n')

# Estimate prediction interval for entire test data
Pred4 <- deepGLMpredict(deepGLMout,XTest,y=yTest,Interval=1,Nsample=1000)
y_pred <- colMeans(Pred4$yhatMatrix)
mse2 <- mean((yTest-y_pred)^2)
accuracy <- (yTest<Pred4$interval[,2] & yTest>Pred4$interval[,1])
cat('Prediction Interval accuracy: ',sum(accuracy)/length(accuracy),'\n')



