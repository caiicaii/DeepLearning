# Part 1 - Building the CNN

# Imports
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialize CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(64, (3, 3), input_shape=(128, 128, 3), activation='relu'))

# Step 2 - Pooling
classifier.add(MaxPool2D(pool_size=(2, 2)))

# Adding second convolutional layer
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPool2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Part 2 - Fitting CNN to images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory('Deep_Learning_A_Z/Volume 1 - Supervised Deep Learning/Part 2 - Convolutional Neural Networks (CNN)/Section 8 - Building a CNN/Convolutional_Neural_Networks/dataset/training_set',
                                                 target_size=(128, 128),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('Deep_Learning_A_Z/Volume 1 - Supervised Deep Learning/Part 2 - Convolutional Neural Networks (CNN)/Section 8 - Building a CNN/Convolutional_Neural_Networks/dataset/test_set',
                                            target_size=(128, 128),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=8000/32,
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=2000/32,
                         workers=8,
                         use_multiprocessing=True)

# Part 3 - Making single predictions
import numpy as np
from keras.preprocessing import image
test_image_1 = image.load_img(path='Deep_Learning_A_Z/Volume 1 - Supervised Deep Learning/Part 2 - Convolutional Neural Networks (CNN)/Section 8 - Building a CNN/Convolutional_Neural_Networks/dataset/single_prediction/cat_or_dog_1.jpg', target_size=(128, 128))
test_image_1 = image.img_to_array(test_image_1)
test_image_1 = np.expand_dims(test_image_1, axis=0)
result_1 = classifier.predict(test_image_1)
training_set.class_indices
if result_1[0][0] == 1:
    prediction_1 = 'dog'
else:
    prediction_1 = 'cat'

test_image_2 = image.load_img(path='Deep_Learning_A_Z/Volume 1 - Supervised Deep Learning/Part 2 - Convolutional Neural Networks (CNN)/Section 8 - Building a CNN/Convolutional_Neural_Networks/dataset/single_prediction/cat_or_dog_2.jpg', target_size=(128, 128))
test_image_2 = image.img_to_array(test_image_2)
test_image_2 = np.expand_dims(test_image_2, axis=0)
result_2 = classifier.predict(test_image_2)
if result_2[0][0] == 1:
    prediction_2 = 'dog'
else:
    prediction_2 = 'cat'

