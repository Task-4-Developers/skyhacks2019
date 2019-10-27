import csv
from keras.models import load_model
import os
import logging
import random
import numpy as np
import copy
import pandas as pd
import pickle
from typing import Tuple
import time
from PIL import Image

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Activation

import numpy as np
from PIL import Image

def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     input_shape=(512, 512, 3)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(47, activation='sigmoid'))
    # compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.load_weights('weights-02-0.2155-bigger.hdf5')

    return model


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

__author__ = 'ING_DS_TECH'
__version__ = "201909"

FORMAT = '%(asctime)-15s %(levelname)s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logger = logging.getLogger(__name__)

input_dir = "test_data2/"
answers_file = "answers.csv"

labels_task_1 = ['Bathroom', 'Bathroom cabinet', 'Bathroom sink', 'Bathtub', 'Bed', 'Bed frame',
                 'Bed sheet', 'Bedroom', 'Cabinetry', 'Ceiling', 'Chair', 'Chandelier', 'Chest of drawers',
                 'Coffee table', 'Couch', 'Countertop', 'Cupboard', 'Curtain', 'Dining room', 'Door', 'Drawer',
                 'Facade', 'Fireplace', 'Floor', 'Furniture', 'Grass', 'Hardwood', 'House', 'Kitchen',
                 'Kitchen & dining room table', 'Kitchen stove', 'Living room', 'Mattress', 'Nightstand',
                 'Plumbing fixture', 'Property', 'Real estate', 'Refrigerator', 'Roof', 'Room', 'Rural area',
                 'Shower', 'Sink', 'Sky', 'Table', 'Tablecloth', 'Tap', 'Tile', 'Toilet', 'Tree', 'Urban area',
                 'Wall', 'Window']

labels_task2 = ['apartment', 'bathroom', 'bedroom', 'dinning_room', 'house', 'kitchen', 'living_room']

labels_task3_1 = [1, 2, 3, 4]
labels_task3_2 = [1, 2, 3, 4]

output = []

test1_model = define_model()

def find(name, path):
    for root, _, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

def task_1(partial_output: dict, file_path: str) -> dict:
    logger.debug("Performing Task 1 for file {0}".format(file_path))

    for label in labels_task_1:
        partial_output[label] = 0 

    image = Image.open(file_path)
    input_array = [np.array(image)]

    prediction = test1_model.predict(np.array(input_array))
    print("Model1 prediction:")
    print(prediction[0].dtype)
    print(prediction[0])

    output_values = []
    for value in prediction[0]:
        if value > 0.8:
            output_values.append(1)
        else:
            output_values.append(0)

    labels = copy.deepcopy(labels_task_1)
    labels.remove('Bathroom')
    labels.remove('Bedroom')
    labels.remove('Dining room')
    labels.remove('House')
    labels.remove('Kitchen')
    labels.remove('Living room')

    print("Labels length: ")
    print(len(labels))
    print("Output values length: ")
    print(len(output_values))

    index = 0
    for label in labels:
        partial_output[label] = output_values[index]
        index = index + 1

    print("Partial output:")
    print(partial_output)

    #
    #
    #	HERE SHOULD BE A REAL SOLUTION
    #
    #
    logger.debug("Done with Task 1 for file {0}".format(file_path))
    return partial_output


def task_2(partial_output: dict, file_path: str) -> str:
    logger.debug("Performing Task 2 for file {0}".format(file_path))
    #
    #
    #	HERE SHOULD BE A REAL SOLUTION
    #
    #
    # Open trained model
    with open("task2_model.pkl", 'rb') as file:
        model = pickle.load(file)

        adjusted_input = copy.deepcopy(partial_output)

        adjusted_input.pop('filename', None)
        adjusted_input.pop('Bathroom', None)
        adjusted_input.pop('Bedroom', None)
        adjusted_input.pop('Living room', None)
        adjusted_input.pop('Kitchen', None)
        adjusted_input.pop('Dining room', None)
        adjusted_input.pop('House', None)

        for key in adjusted_input:
            adjusted_input[key] = [adjusted_input[key]]



        predicted_class = model.predict(pd.DataFrame.from_dict(adjusted_input))

        if predicted_class[0] == 'living_room':
            partial_output['Living room'] = 1
        if predicted_class[0] == 'kitchen':
            partial_output['Kitchen'] = 1
        if predicted_class[0] == 'house':
            partial_output['House'] = 1
        if predicted_class[0] == 'dining_room':
            partial_output['Dining room'] = 1
        if predicted_class[0] == 'bedroom':
            partial_output['Bedroom'] = 1
        if predicted_class[0] == 'bathroom':
            partial_output['Bathroom'] = 1

        if predicted_class[0] == 'dining_room':
            predicted_class[0] = 'dinning_room'

        logger.debug("Done with Task 2 for file {0}".format(file_path))
        return predicted_class[0]
    
    logger.debug("Failed with Task 2 for file {0}".format(file_path))
    return "failed"


def task_3(file_path: str) -> Tuple[str, str]:
    logger.debug("Performing Task 3 for file {0}".format(file_path))
    #
    #
    #	HERE SHOULD BE A REAL SOLUTION
    #
    #
    logger.debug("Done with Task 3 for file {0}".format(file_path))
    return labels_task3_1[random.randrange(len(labels_task3_1))], labels_task3_2[random.randrange(len(labels_task3_2))]


def main():
    logger.debug("Sample answers file generator")
    for dirpath, dnames, fnames in os.walk(input_dir):
        for f in fnames:
            if f.endswith(".jpg"):
                file_path = os.path.join(dirpath, f)
                output_per_file = task_1({'filename': f}, file_path)
                output_per_file['task2_class'] = task_2(output_per_file, file_path)
                output_per_file['tech_cond'] = task_3(file_path)[0]
                output_per_file['standard'] = task_3(file_path)[1]

                output.append(output_per_file)


    with open(answers_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['filename', 'standard', 'task2_class', 'tech_cond'] + labels_task_1)
        writer.writeheader()
        for entry in output:
            logger.debug(entry)
            writer.writerow(entry)


if __name__ == "__main__":
    main()
