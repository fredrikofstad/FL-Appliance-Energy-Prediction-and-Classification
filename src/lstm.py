import tensorflow as tf


class LSTM (tf.keras.Sequential):
    def __init__(self):
        super(LSTM, self).__init__()

    def build_default(self, input_shape, n_classes, use_batch_shape=False, activation="softmax", verbose=0):
        model = tf.keras.models.Sequential()
        if use_batch_shape:
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True),
                                                    batch_input_shape=input_shape))
        else:
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True),
                                                    input_shape=input_shape))
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512)))
            model.add(tf.keras.layers.Dense(n_classes, activation=activation))
            if verbose:
                model.summary()
            return model
