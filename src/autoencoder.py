import tensorflow as tf
import csv

import os
import time
import numpy as np
from keras.models import Model, Sequential, clone_model
from keras.layers import Input, Dense, Activation, Concatenate
from keras.utils import Sequence
from keras import backend as K



autoencoderLayers = [773, 600, 400, 200, 100]

dbnLayers = len(autoencoderLayers) - 1

def test_autoencoder():
    for i in range(dbnLayers):

        dbn_model = Sequential()
        dbn_model.add(Dense(autoencoderLayers[i+1], activation='relu',
                            input_dim=autoencoderLayers[i]))
        dbn_model.add(Dense(autoencoderLayers[i], activation='relu'))
        dbn_model.compile(optimizer='adam',
                        loss='mse',
                        metrics=['accuracy'])

        dbn_model.summary()

        # """ TRAIN MODEL """
        # dbn_model.fit(mat, mat, batch_size=batch_size, epochs=dnb_epochs)
        # # score = dbn_model.evaluate(mat, mat, batch_size=batch_size)
        # # print(score)

        # # GET THE WEIGHT MATRIX
        # weightMatrix.append(dbn_model.layers[0].get_weights())
        # # Get the outputs of the hidden layer
        # getHiddenOuptut = K.function(
        #     [dbn_model.input], [dbn_model.layers[0].output])
        # mat = getHiddenOuptut([mat])[0]
        # # print("HIDDEN SHAPE: ", getHiddenOuptut)
        # # print(weightMatrix)
        # # shape_vec.append(mat.shape)


        # # print("WEIGHT MATRIX SHAPE:", len(
        # #     weightMatrix[0][0]), len(weightMatrix[0][0][0]))
        # # print("WEIGHT MATRIX SHAPE:", len(
        # #     weightMatrix[1][0]), len(weightMatrix[1][0][0]))
        # # print("WEIGHT MATRIX SHAPE:", len(
        # #     weightMatrix[2][0]), len(weightMatrix[2][0][0]))
        # # print("WEIGHT MATRIX SHAPE:", len(
        # #     weightMatrix[3][0]), len(weightMatrix[3][0][0]))


def create_autoencoder():
    """
    Creates an autoencoder 69->40->20->(10)->20->40->69
    
    Returns the full autoencoder model as well as a model of the section that encodes
    """
    input_board = tf.keras.Input(shape=(69,))
    encoder_layer_1 = tf.keras.layers.Dense(40, activation='relu')(input_board)
    encoder_layer_2 = tf.keras.layers.Dense(20, activation='relu')(encoder_layer_1)
    encoded = tf.keras.layers.Dense(10, activation='relu')(encoder_layer_2)
    decoder_layer_1 = tf.keras.layers.Dense(20, activation='relu')(encoded)
    decoder_layer_2 = tf.keras.layers.Dense(40, activation='relu')(decoder_layer_1)
    decoded = tf.keras.layers.Dense(69, activation='relu')(decoder_layer_2)

    autoencoder = tf.keras.Model(input_board, decoded)

    return autoencoder



def train_encoder(autoencoder, training_set, testing_set):
    """
    Takes an autoencoder, a training data set of chess boards and a testing set.
    
    Fits the chess board autoencoder based on the training set and tests with the separate testing set.

    Runs 100 epochs with shuffling on. 

    Finally, saves the autoencoder.
    """
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    autoencoder.fit(x=training_set, y=training_set, epochs=50, shuffle=True, validation_data=(testing_set, testing_set))
    autoencoder.save("saved_networks/autoencoder_model")


def setup_autoencoder():
    autoencoder = create_autoencoder()
    all_data = []
    with open('test.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',',quotechar='"')
        firstline = True
        for row in reader:
            if firstline:
                firstline = False
                continue

            all_data.append([int(tile) for tile in row[-1][1:-1].split(', ')])
    
    training_data = all_data[:len(all_data) // 2]
    testing_data = all_data[len(all_data) // 2 + 1:]
    train_encoder(autoencoder, training_data, testing_data)


if __name__ == "__main__":
    #setup_autoencoder()
    test_autoencoder()
