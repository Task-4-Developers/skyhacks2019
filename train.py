from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

import pandas as pd
from PIL import Image


file = 'dataset_github/2019/main/labels.csv'
names = ['filename','standard','task2_class','tech_cond','Bathroom','Bathroom cabinet','Bathroom sink','Bathtub','Bed','Bed frame','Bed sheet','Bedroom','Cabinetry','Ceiling','Chair','Chandelier','Chest of drawers','Coffee table','Couch','Countertop','Cupboard','Curtain','Dining room','Door','Drawer','Facade','Fireplace','Floor','Furniture','Grass','Hardwood','House','Kitchen','Kitchen & dining room table','Kitchen stove','Living room','Mattress','Nightstand','Plumbing fixture','Property','Real estate','Refrigerator','Roof','Room','Rural area','Shower','Sink','Sky','Table','Tablecloth','Tap','Tile','Toilet','Tree','Urban area','Wall','Window']
labels_data = pd.read_csv(file, names=names)
to_detect = 'Bed'


labels_data = labels_data.drop(labels_data.index[0])

import os

def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

"""
for filename in labels_data['filename']:
    file_path = find(filename, "resized_images/")

    #do something with the file
    print(file_path)
    file = open(file_path, "r")
    file.close()
"""
def get_data():
    global to_detect
    global labels_data
    images = []
    labels = []
    for filename in labels_data['filename']:
        file_path = find(filename, "resized_images/")
        print(file_path)
        try:
            tmp = Image.open(file_path)
            images.append(tmp)
            if filename == to_detect:
                labels.append(1)
            else:
                labels.append(0)
        except:
	        pass
    return images, labels


def train_network():
    network_input, network_output = get_data()

    # get amount of pitch names
    #n_vocab = len(set(notes))

    # network_input, network_output = prepare_sequences(notes, n_vocab)

    model = define_model() #create_network(network_input, n_vocab)

    train(model, network_input, network_output)


def train_test_set(dataset, ratio_test, ratio_true):
    return dataset



def create_network(network_input, input_shape):
    model = Sequential()
    model.add(Conv2D(32, (2, 2), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


def train(model, network_input, network_output):
    """ train the neural network """
    global to_detect
    filepath = "weights-{to_detect}/{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=20, batch_size=64, callbacks=callbacks_list)


def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


train_network()