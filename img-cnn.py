
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt


# load data


train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

train_0_dir = os.path.join(train_dir, '00')  # directory with our training pictures
train_1_dir = os.path.join(train_dir, '01')  # directory with our training dog pictures
validation_0_dir = os.path.join(validation_dir, '0')  # directory with our validation  pictures
validation_dogs_dir = os.path.join(validation_dir, '1')  # directory with our validation  pictures


# understand the data


num_00_tr = len(os.listdir(train_0_dir))
num_01_tr = len(os.listdir(train_1_dir))

num_00_val = len(os.listdir(validation_0_dir))
num_01_val = len(os.listdir(validation_1_dir))

total_train = num_00_tr + num_01_tr
total_val = num_c00_val + num_01_val

print('total training  images:', num_00_tr)
print('total training  images:', num_01_tr)

print('total validation  images:', num_00_val)
print('total validation  images:', num_01_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)


batch_size = 128
epochs = 15
IMG_HEIGHT = 500
IMG_WIDTH = 500


# data preparation


train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
															directory=train_dir,
															shuffle=True,
															target_size=(IMG_HEIGHT, IMG_WIDTH),
															class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
															directory=validation_dir,
															target_size=(IMG_HEIGHT, IMG_WIDTH),
															class_mode='binary')


# create model


model = Sequential([
	Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
	MaxPooling2D(),
	Conv2D(32, 3, padding='same', activation='relu'),
	MaxPooling2D(),
	Conv2D(64, 3, padding='same', activation='relu'),
	MaxPooling2D(),
	Flatten(),
	Dense(512, activation='relu'),
	Dense(1)
])



model.compile(optimizer='adam',
				loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
				metrics=['accuracy'])
model.summary()

history = model.fit_generator(
	train_data_gen,
	steps_per_epoch=total_train // batch_size,
	epochs=epochs,
	validation_data=val_data_gen,
	validation_steps=total_val // batch_size
)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

