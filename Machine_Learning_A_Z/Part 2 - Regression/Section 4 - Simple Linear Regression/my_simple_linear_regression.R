# Simple Linear Regression

# Importing dataset
dataset = read.csv('~/Desktop/DeepLearning/Machine Learning A-Z New/Part 2 - Regression/Section 4 - Simple Linear Regression/Salary_Data.csv')

# Splitting into Training and Test set
# install.packages(caTools)
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Simple Linear Regression to Training set
regressor = lm(formula = Salary ~ YearsExperience, 
               data = training_set)

# Predicting Test set results
y_pred = predict(regressor, newdata = test_set)

# Visualizing Training set results
# install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Training set)') +
  xlab('Years of Experience') +
  ylab('Salary')

# Visualizing Test set results
library(ggplot2)
ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Test set)') +
  xlab('Years of Experience') +
  ylab('Salary')