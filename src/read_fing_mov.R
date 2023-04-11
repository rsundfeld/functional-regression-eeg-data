require(reticulate)
require(dplyr)
library(refund)
library(mgcv)
library(pROC)
library(caret)
np <- import('numpy')

#1. reading the data that was created in the Python Notebook
x_train_py <- np$load('C:\\Users\\Rodolfo\\Documents\\Rodolfo\\USP\\dissertacao\\datasets\\Finger Movements\\Datasets_clean\\FingerMovements\\X_train.npy')
x_test_py <- np$load('C:\\Users\\Rodolfo\\Documents\\Rodolfo\\USP\\dissertacao\\datasets\\Finger Movements\\Datasets_clean\\FingerMovements\\X_test.npy')

y_train_py <- np$load('C:\\Users\\Rodolfo\\Documents\\Rodolfo\\USP\\dissertacao\\datasets\\Finger Movements\\Datasets_clean\\FingerMovements\\y_train.npy')
y_test_py <- np$load('C:\\Users\\Rodolfo\\Documents\\Rodolfo\\USP\\dissertacao\\datasets\\Finger Movements\\Datasets_clean\\FingerMovements\\y_test.npy')

x_train_melted <- x_train_py
x_test_melted <- x_test_py

#2. setting data dimensions (316/100 train/test observations of 28 EEG channels with 50 time points each)
trials_train = 316
trials_test = 100
channels = 28
time_points = 50

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

#3. creating target
y_train <- ifelse(y_train_py == 'right', 1, 0)
y_test <- ifelse(y_test_py == 'right', 1, 0)

#4. resampling what's train and what's test randomly
x <- rbind(x_train, x_test)
y <- c(y_train, y_test) 

#set.seed(10); train_ids = sample(1:nrow(x), size = floor(nrow(x)*0.8))
#x_train = x[train_ids,]; y_train = y[train_ids]
#x_test = x[-train_ids,]; y_test = y[-train_ids]