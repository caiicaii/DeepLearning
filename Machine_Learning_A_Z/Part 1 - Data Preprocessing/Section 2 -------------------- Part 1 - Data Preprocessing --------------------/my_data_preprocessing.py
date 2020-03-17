# Data Preprocessing

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import dataset
dataset = pd.read_csv(
    'Machine Learning A-Z New/Part 1 - Data Preprocessing/Section 2 -------------------- Part 1 - Data Preprocessing --------------------/Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# # Missing data
# from sklearn.impute import SimpleImputer
#
# imputer = SimpleImputer(missing_values=np.nan, strategy='mean', verbose=0)
# imputer = imputer.fit(X[:, 1:3])
# X[:, 1:3] = imputer.transform(X[:, 1:3])
#
# # Encode categorical data
# from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# from sklearn.compose import ColumnTransformer
#
# ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# X = np.array(ct.fit_transform(X), dtype=np.float)
#
# y = LabelEncoder().fit_transform(y)

# Splitting dataset into Training and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# # Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.fit_transform(X_test)
# sc_y = StandardScaler()
# y_train = sc_y.fit_transform(y_train)
