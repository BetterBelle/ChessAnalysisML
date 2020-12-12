import tensorflow as tf
import autoencoder as ac
import numpy as np

def create_deepchess():
    autoencoder = tf.keras.models.load_model("saved_networks/autoencoder_model")

    right_encoder = tf.keras.Model(inputs=autoencoder.input, outputs=autoencoder.layers[-4].output)
    left_encoder= tf.keras.models.clone_model(right_encoder)
    left_encoder.set_weights(right_encoder.get_weights())

    for i, layer in enumerate(right_encoder.layers):
        layer._name = 'right_encoder_layer_' + str(i)
    for i, layer in enumerate(left_encoder.layers):
        layer._name = 'left_encoder_layer_' + str(i)

    right_encoder.compile(optimizer='adam', loss='binary_crossentropy')
    left_encoder.compile(optimizer='adam', loss='binary_crossentropy')

    print("Right Summary")
    right_encoder.summary()
    print("Left Summary")
    left_encoder.summary()

    combined = tf.keras.layers.concatenate([left_encoder.output, right_encoder.output])

    chess_network = tf.keras.layers.Dense(40, activation='relu')(combined)
    chess_network = tf.keras.layers.Dense(20, activation='relu')(chess_network)
    chess_network = tf.keras.layers.Dense(10, activation='relu')(chess_network)
    chess_network = tf.keras.layers.Dense(2, activation='relu', name='dense_3')(chess_network)

    chess_model = tf.keras.Model(inputs=[left_encoder.input, right_encoder.input], outputs=chess_network)
    chess_model.compile(optimizer='adam', loss='binary_crossentropy')
    print("Deep Chess Summary")
    chess_model.save('saved_networks/deepchess_model')
    chess_model.summary()


def train_deepchess(deepchess_model, training_set):
    for i, layer in enumerate(deepchess_model.layers):
        if i == 8:
            break
        layer.trainable = False
    deepchess_model.summary()
    deepchess_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    deepchess_model.fit(
        {
            'left_encoder_layer_0': np.array(training_set[0]), 
            'right_encoder_layer_0': np.array(training_set[1])
        },
        {
            'dense_3': np.array(training_set[2])
        }, 
        epochs=50, 
        shuffle=True
    )
    deepchess_model.save("saved_networks/autoencoder_model")
