import tensorflow as tf


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
    encoder = tf.keras.Model(input_board, encoded)

    return autoencoder, encoder



def train_encoder(autoencoder, encoder, training_set, testing_set):
    """
    Takes an autoencoder, the encoding part of the autoencoder, a training data set of chess boards and a testing set.
    
    Fits the chess board autoencoder based on the training set and tests with the separate testing set.

    Runs 50 epochs of batch sizes of 100 with shuffling on. 

    Finally, saves the encoder only.
    """
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    # encoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.fit(training_set, training_set, epochs=50, batch_size=100, shuffle=True, validation_data=(testing_set, testing_set))
    autoencoder.save("saved_networks/encoder_model")