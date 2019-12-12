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

# Fitting Multiple Linear Regression to Training set
regressor = lm(formula = Profit ~ .,
               data = training_set)

# Predicting Test set results
y_pred = predict(regressor, newdata = test_set)