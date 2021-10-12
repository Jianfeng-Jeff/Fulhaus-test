'''
Test the funiture classifier


'''
# import useful libraies
import os
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Flatten,MaxPooling2D,Dropout
from keras.models import load_model


# preset parameters
IMAGE_SHAPE = (339, 376, 3)
BATCH_SIZE = 4

# load model
model = load_model('furniture_classifier.h5')
# summarize model.
model.summary()
print("Model loaded!")

class_labels = {'Bed': 0, 'Chair': 1, 'Sofa': 2}


test_path = 'Data_for_test'

test_image_gen = ImageDataGenerator(rescale=1./255)   

test_generator = test_image_gen.flow_from_directory(directory=test_path,
                                                 target_size=IMAGE_SHAPE[:2],
                                                 color_mode='rgb',
                                                 batch_size=BATCH_SIZE,
                                                 class_mode=None, shuffle=False)
                                                
pred = model.predict(test_generator,steps=len(test_generator),verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)

labels = (class_labels)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]


# save prediction result and ground truth to csv file
filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})

def file_name(str):
    folder_name = str.split('/')
    return folder_name[0]

results['Ground Truth'] = results['Filename'].apply(file_name)


results['Prediction Correctness'] = np.where(results['Predictions']==results['Ground Truth'], True, False)
print(results)

number_of_rows = len(results.index)

prediction_accuracy = (results['Prediction Correctness']).values.sum()/number_of_rows
print('The accuracy of testing dataset is: ', prediction_accuracy)


results.to_csv('Furniture classification.csv',index=True)
print('Result saving to csv file')
