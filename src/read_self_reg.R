require(reticulate)
require(dplyr)
library(refund)
library(mgcv)
library(pROC)
library(caret)
np <- import('numpy')

#1. reading the data that was created in the Python Notebook
x_train_py <- np$load('C:\\Users\\Rodolfo\\Documents\\Rodolfo\\USP\\dissertacao\\datasets\\Self Regulation\\Datasets_clean\\SelfRegulationSCP1\\X_train.npy')
x_test_py <- np$load('C:\\Users\\Rodolfo\\Documents\\Rodolfo\\USP\\dissertacao\\datasets\\Self Regulation\\Datasets_clean\\SelfRegulationSCP1\\X_test.npy')

y_train_py <- np$load('C:\\Users\\Rodolfo\\Documents\\Rodolfo\\USP\\dissertacao\\datasets\\Self Regulation\\Datasets_clean\\SelfRegulationSCP1\\y_train.npy')
y_test_py <- np$load('C:\\Users\\Rodolfo\\Documents\\Rodolfo\\USP\\dissertacao\\datasets\\Self Regulation\\Datasets_clean\\SelfRegulationSCP1\\y_test.npy')

x_train_melted <- x_train_py
x_test_melted <- x_test_py

trials_train = 268
trials_test = 293
channels = 6
time_points = 896

dim(x_train_melted) <- c(trials_train*channels, time_points)
dim(x_test_melted) <- c(trials_test*channels, time_points)

x_train <- data.frame(id = 1:trials_train)
for(i in 1:channels){
  x_train[,(1+i)] <- x_train_melted[(1+(i-1)*trials_train):(trials_train+(i-1)*trials_train),]
}
colnames(x_train) = c("id", sprintf("V%02d", seq(1,channels)))

x_test <- data.frame(id = 1:trials_test)
for(i in 1:channels){
  x_test[,(1+i)] <- x_test_melted[(1+(i-1)*trials_test):(trials_test+(i-1)*trials_test),]
}
colnames(x_test) = c("id", sprintf("V%02d", seq(1,channels)))

y_train <- ifelse(y_train_py == 'positivity', 1, 0)
y_test <- ifelse(y_test_py == 'positivity', 1, 0)

x <- rbind(x_train, x_test)
y <- c(y_train, y_test) 