# Multiple Linear Regression

# Importing Dataset
dataset = read.csv('~/Desktop/DeepLearning/Machine Learning A-Z New/Part 2 - Regression/Section 5 - Multiple Linear Regression/50_Startups.csv')

# Encode categorical data
dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1, 2, 3))

# Split the dataset into Training and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)