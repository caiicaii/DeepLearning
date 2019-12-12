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

# Optimizing model with Backward Elimination
regressor = lm(formula = Profit ~ + R.D.Spend + Administration + Marketing.Spend + State,
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ + R.D.Spend + Administration + Marketing.Spend,
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ + R.D.Spend + Marketing.Spend,
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ + R.D.Spend,
               data = dataset)
summary(regressor)

# Automatic Backward Elimination
backwardElimination <- function(x, sl) {
  num_vars = length(x)
  for (i in c(1:num_vars)) {
    regressor = lm(formula = Profit ~ ., data = x)
    max_var = max(coef(summary(regressor))[c(2:num_vars), "Pr(>|t|)"])
    
    if (max_var > sl) {
      j = which(coef(summary(regressor))[c(2:num_vars), "Pr(>|t|)"] == max_var)
      x = x[, -j]
    }
    num_vars = num_vars - 1
  }
  return(summary(regressor))
}

SL = 0.05
dataset = dataset[, c(1, 2, 3, 4, 5)]
backwardElimination(training_set, SL)
