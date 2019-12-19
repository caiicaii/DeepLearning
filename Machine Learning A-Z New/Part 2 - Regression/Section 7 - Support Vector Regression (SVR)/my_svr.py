# Support Vector Regression (SVR)

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing Dataset
dataset = pd.read_csv('Machine Learning A-Z New/Part 2 - Regression/Section 7 - Support Vector Regression (SVR)/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = np.squeeze(sc_y.fit_transform(y.reshape(-1, 1)))

# Fitting SVR to dataset
from sklearn.svm import SVR
regressor = SVR()
regressor.fit(X, y)

# Predicting new result
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

# Visualizing the SVR results
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualizing SVR results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()