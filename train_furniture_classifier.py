'''
Train a funiture classifier


'''
# import useful libraies
import os
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Flatten,MaxPooling2D,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint


# Training Data directory
DATA_PATH = '/home/jianfeng/Desktop/Fulhaus_docker/furniture_dataset'
TRAIN_PATH = os.path.join(DATA_PATH, 'train')

print('Class labels:', os.listdir(TRAIN_PATH))


# Preset Parameters
IMAGE_SHAPE = (339, 376, 3)
BATCH_SIZE = 8
epoch = 30

'''
Start of

Data pre-processing and data augmentation
'''

# generate images using the data generator
image_gen = ImageDataGenerator(rotation_range=20, width_shift_range=0.10, height_shift_range=0.10, rescale=1./255, shear_range=0.1, zoom_range=0.1, horizontal_flip=True, fill_mode='nearest', vertical_flip=False, validation_split=0.3)  

image_gen.flow_from_directory(TRAIN_PATH)

'''
End of

Data pre-processing and data augmentation
'''


'''
Start of

train a funiture classifier
'''
# clear the session
keras.backend.clear_session()
np.random.seed(42)

# create a sequential model
model = Sequential()

# convolutional and max pool layer
model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',strides=(1,1),
                activation='relu',input_shape=IMAGE_SHAPE))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',strides=(1,1),
                activation='relu',input_shape=IMAGE_SHAPE))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same',strides=(1,1),
                activation='relu',input_shape=IMAGE_SHAPE))
model.add(MaxPooling2D(pool_size=(2,2)))

# flatten the layer before feeding into the dense layer
model.add(Flatten())

# dense layer together with dropout to prevent overfitting
model.add(Dense(units=128,activation='relu',kernel_initializer='he_normal'))
model.add(Dense(units=64,activation='relu',kernel_initializer='he_normal'))
model.add(Dense(units=32,activation='relu',kernel_initializer='he_normal'))
model.add(Dropout(0.5))

# there are 3 classes
model.add(Dense(units=3,activation='softmax'))

# compile the model #
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ["accuracy"])

# check the model summary # 
model.summary()

# prepare data for train and validation
train_image_gen = image_gen.flow_from_directory(TRAIN_PATH,target_size=IMAGE_SHAPE[:2], color_mode='rgb',batch_size=BATCH_SIZE, class_mode='categorical',seed=1,subset='training')

validation_image_gen = image_gen.flow_from_directory(TRAIN_PATH,target_size=IMAGE_SHAPE[:2], color_mode='rgb',batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False,subset='validation', seed=1)

# check the class indices #
train_image_gen.class_indices

# save the best model
checkpoint_filepath = '/tmp/checkpoint'
model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, monitor='val_accuracy', mode='max', save_best_only=True)

history=model.fit(train_image_gen, validation_data = validation_image_gen, epochs = epoch, callbacks=[model_checkpoint_callback])

'''
End of

train a funiture classifier
'''


# plot the loss and accuracy of the train and validation data
df_loss = pd.DataFrame(model.history.history)
df_loss.head()

df_loss[['loss','accuracy','val_loss','val_accuracy']].plot()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.savefig('training and validation accuracy.png')

model.evaluate(validation_image_gen)

# save the model
model.save("furniture_classifier.h5")
print("Model Saved!")

