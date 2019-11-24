# Imports
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator

# Variables
filters = 32
input_size = (150, 150)
batch_size = 32
dropout_rate = 0.6
epochs = 100

# Initialization
classifier = Sequential()

# Convolution
classifier.add(Conv2D(filters, (3, 3), padding='same', input_shape=(*input_size, 3), activation='relu'))
classifier.add(MaxPool2D(pool_size=(2, 2)))
classifier.add(Conv2D(filters, (3, 3), padding='same', activation='relu'))
classifier.add(MaxPool2D(pool_size=(2, 2)))
classifier.add(Conv2D(filters*2, (3, 3), padding='same', activation='relu'))
classifier.add(MaxPool2D(pool_size=(2, 2)))
classifier.add(Conv2D(filters*2, (3, 3), padding='same', activation='relu'))
classifier.add(MaxPool2D(pool_size=(2, 2)))

# Flatten
classifier.add(Flatten())

# Full connection
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dropout(dropout_rate))
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dropout(dropout_rate/2))
classifier.add(Dense(units=1, activation='sigmoid'))

# Compile
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit images
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(
    'Deep_Learning_A_Z/Volume 1 - Supervised Deep Learning/Part 2 - Convolutional Neural Networks (CNN)/Section 8 - Building a CNN/Convolutional_Neural_Networks/dataset/training_set',
    target_size=input_size,
    batch_size=batch_size,
    class_mode='binary')

test_set = test_datagen.flow_from_directory(
    'Deep_Learning_A_Z/Volume 1 - Supervised Deep Learning/Part 2 - Convolutional Neural Networks (CNN)/Section 8 - Building a CNN/Convolutional_Neural_Networks/dataset/test_set',
    target_size=input_size,
    batch_size=batch_size,
    class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=8000/batch_size,
                         epochs=epochs,
                         validation_data=test_set,
                         validation_steps=2000/batch_size,
                         workers=16,
                         use_multiprocessing=True)
