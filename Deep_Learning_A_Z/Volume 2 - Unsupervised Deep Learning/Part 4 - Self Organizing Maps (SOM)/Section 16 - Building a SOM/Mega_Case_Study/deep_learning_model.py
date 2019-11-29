# Case Study - Make a Hybrid Deep Learning Model

# Part 1 - Self-Organizing Map
# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('Deep_Learning_A_Z/Volume 2 - Unsupervised Deep Learning/Part 4 - Self Organizing Maps (SOM)/Section 16 - Building a SOM/Mega_Case_Study/Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)

# Import SOM
# Add path to sys to import
import sys
sys.path.insert(0, 'Deep_Learning_A_Z/Volume 2 - Unsupervised Deep Learning/Part 4 - Self Organizing Maps (SOM)/Section 16 - Building a SOM/Self_Organizing_Maps')
from minisom import MiniSom

# Train the SOM
som = MiniSom(x=10, y=10, input_len=15)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

# Visualization
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         colors[y[i]],
         markeredgecolor=colors[y[i]],
         markerfacecolor='None',
         markersize=10,
         markeredgewidth=2)
show()

# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(3, 1)], mappings[(9, 6)]), axis=0)
frauds = sc.inverse_transform(frauds)


# Part 2 - Supervised Deep Learning
# Creating matrix of features
customers = dataset.iloc[:, 1:].values

# Creating dependent variable
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i, 0] in frauds:
        is_fraud[i] = 1

# Part 3 - Making the ANN
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Initializing ANN
classifier = Sequential()

# Adding input layer and first hidden layer
classifier.add(Dense(units=2, kernel_initializer='uniform', activation='relu', input_shape=(15,)))

# Add output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compile ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit ANN to training set
classifier.fit(customers, is_fraud, batch_size=1, epochs=2)

# Predicting the Test set results
y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1], y_pred), axis=1)
y_pred = y_pred[y_pred[:, 1].argsort()]
