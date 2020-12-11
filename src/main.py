import tensorflow as tf
import csv
import autoencoder as ac


def main():
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

    autoencoder = tf.keras.models.load_model("saved_networks/autoencoder_model")

    right_encoder = tf.keras.Model(inputs=autoencoder.input, outputs=autoencoder.layers[-4].output)
    left_encoder= tf.keras.models.clone_model(right_encoder)
    left_encoder.set_weights(right_encoder.get_weights())

    for i, layer in enumerate(right_encoder.layers):
        layer._name = 'right_encoder_layer' + str(i)
        layer.trainable = False
    for i, layer in enumerate(left_encoder.layers):
        layer._name = 'left_encoder_layer_' + str(i)
        layer.trainable = False

    right_encoder.compile(optimizer='adam', loss='binary_crossentropy')
    left_encoder.compile(optimizer='adam', loss='binary_crossentropy')

    right_encoder.summary()
    left_encoder.summary()

    combined = tf.keras.layers.concatenate([right_encoder.output, left_encoder.output])

    chess_network = tf.keras.layers.Dense(40, activation='relu')(combined)
    chess_network = tf.keras.layers.Dense(20, activation='relu')(chess_network)
    chess_network = tf.keras.layers.Dense(10, activation='relu')(chess_network)
    chess_network = tf.keras.layers.Dense(2, activation='relu')(chess_network)

    chess_model = tf.keras.Model(inputs=[left_encoder.input, right_encoder.input], outputs=chess_network)
    chess_model.compile(optimizer='adam', loss='binary_crossentropy')
    chess_model.summary()


    
if __name__ == "__main__":
    main()