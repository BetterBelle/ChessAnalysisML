import tensorflow as tf
import csv


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
    autoencoder.fit(x=training_set, y=training_set, epochs=100, shuffle=True, validation_data=(testing_set, testing_set))
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
    testing_data = all_data[len(all_data) //2 + 1:]
    train_encoder(autoencoder, training_data, testing_data)


if __name__ == "__main__":
    setup_autoencoder()
