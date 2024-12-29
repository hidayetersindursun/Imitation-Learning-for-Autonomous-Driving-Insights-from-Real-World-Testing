#    This file was created by
#    MATLAB Deep Learning Toolbox Converter for TensorFlow Models.
#    12-Mar-2024 10:50:15

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model():
    imageinput_unnormalized = keras.Input(shape=(672,224,3), name="imageinput_unnormalized")
    imageinput = ZScoreLayer((672,224,3), name="imageinput_")(imageinput_unnormalized)
    conv_1 = layers.Conv2D(24, (5,5), strides=(2,2), name="conv_1_")(imageinput)
    layer_1 = layers.ELU(alpha=1.000000)(conv_1)
    conv_2 = layers.Conv2D(36, (5,5), strides=(2,2), name="conv_2_")(layer_1)
    layer_2 = layers.ELU(alpha=1.000000)(conv_2)
    conv_3 = layers.Conv2D(48, (5,5), strides=(2,2), name="conv_3_")(layer_2)
    layer_3 = layers.ELU(alpha=1.000000)(conv_3)
    conv_4 = layers.Conv2D(64, (5,5), strides=(2,2), name="conv_4_")(layer_3)
    layer_4 = layers.ELU(alpha=1.000000)(conv_4)
    conv_5 = layers.Conv2D(64, (3,3), name="conv_5_")(layer_4)
    layer_5 = layers.ELU(alpha=1.000000)(conv_5)
    conv_6 = layers.Conv2D(64, (3,3), name="conv_6_")(layer_5)
    layer_6 = layers.ELU(alpha=1.000000)(conv_6)
    dropout = layers.Dropout(0.500000)(layer_6)
    dropoutperm = layers.Permute((3,2,1))(dropout)
    flatten_1 = layers.Flatten()(dropoutperm)
    layer_7 = layers.ELU(alpha=1.000000)(flatten_1)
    fc_1 = layers.Dense(100, name="fc_1_")(layer_7)
    layer_8 = layers.ELU(alpha=1.000000)(fc_1)
    fc_2 = layers.Dense(50, name="fc_2_")(layer_8)
    layer_9 = layers.ELU(alpha=1.000000)(fc_2)
    fc_3 = layers.Dense(10, name="fc_3_")(layer_9)
    flatten_2 = layers.Flatten()(fc_3)
    flatten_2_lstm_input = flatten_2
    flatten_2_lstm_input = layers.Reshape((1,-1))(flatten_2_lstm_input)
    lstm1 = layers.LSTM(64, name='lstm1_', activation='tanh', recurrent_activation='sigmoid', return_sequences=False, return_state=False)(flatten_2_lstm_input)
    lstm1 = layers.Reshape((-1,))(lstm1)
    fc_4 = layers.Dense(1, name="fc_4_")(lstm1)
    regression = fc_4

    model = keras.Model(inputs=[imageinput_unnormalized], outputs=[regression])
    return model

## Helper layers:

class ZScoreLayer(tf.keras.layers.Layer):
    def __init__(self, shape, name=None):
        super(ZScoreLayer, self).__init__(name=name)
        self.mean = tf.Variable(initial_value=tf.zeros(shape), trainable=False)
        self.std = tf.Variable(initial_value=tf.zeros(shape), trainable=False)

    def call(self, input):
        # Compute z-score of input
        return (input - self.mean)/self.std

