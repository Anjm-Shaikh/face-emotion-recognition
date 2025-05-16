import os
import cv2
import numpy as np
from keras.models import Sequential 

from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.utils import to_categorical
from keras.optimizers import Adam

# Data path
train_dir = 'data/train'
test_dir = 'data/test'

# Image size
img_size = 64
batch_size = 64

# Data augmentation
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=30, zoom_range=0.2,
                                   width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(img_size, img_size),
    color_mode='grayscale', batch_size=batch_size,
    class_mode='categorical', shuffle=True)

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(img_size, img_size),
    color_mode='grayscale', batch_size=batch_size,
    class_mode='categorical', shuffle=False)

# Build CNN model
model = Sequential()

model.add(Conv2D(64, (3,3), activation='relu', input_shape=(img_size, img_size, 1)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))  # 7 emotion classes

# Compile model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(train_generator, validation_data=test_generator, epochs=25)

# Save model
model.save('models/emotion_model.hdf5')
print("✅ Model saved as emotion_model.hdf5")
