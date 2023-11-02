#    This file was created by
#    MATLAB Deep Learning Toolbox Converter for TensorFlow Models.
#    24-Oct-2023 14:14:38

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model():
    imageinput_unnormalized = keras.Input(shape=(224,224,3), name="imageinput_unnormalized")
    imageinput = SubtractConstantLayer((224,224,3), name="imageinput_")(imageinput_unnormalized)
    conv_1 = layers.Conv2D(8, (3,3), padding="same", name="conv_1_")(imageinput)
    batchnorm_1 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm_1_")(conv_1)
    relu_1 = layers.ReLU()(batchnorm_1)
    avgpool2d_1 = layers.AveragePooling2D(pool_size=(2,2), strides=(2,2))(relu_1)
    conv_2 = layers.Conv2D(16, (3,3), padding="same", name="conv_2_")(avgpool2d_1)
    batchnorm_2 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm_2_")(conv_2)
    relu_2 = layers.ReLU()(batchnorm_2)
    avgpool2d_2 = layers.AveragePooling2D(pool_size=(2,2), strides=(2,2))(relu_2)
    conv_3 = layers.Conv2D(32, (3,3), padding="same", name="conv_3_")(avgpool2d_2)
    batchnorm_3 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm_3_")(conv_3)
    relu_3 = layers.ReLU()(batchnorm_3)
    conv_4 = layers.Conv2D(32, (3,3), padding="same", name="conv_4_")(relu_3)
    batchnorm_4 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm_4_")(conv_4)
    relu_4 = layers.ReLU()(batchnorm_4)
    dropout = layers.Dropout(0.200000)(relu_4)
    fc = layers.Reshape((1, 1, -1), name="fc_preFlatten1")(dropout)
    fc = layers.Dense(1, name="fc_")(fc)
    regressionoutput = layers.Flatten()(fc)

    model = keras.Model(inputs=[imageinput_unnormalized], outputs=[regressionoutput])
    return model

## Helper layers:

class SubtractConstantLayer(tf.keras.layers.Layer):
    def __init__(self, shape, name=None):
        super(SubtractConstantLayer, self).__init__(name=name)
        self.const = tf.Variable(initial_value=tf.zeros(shape), trainable=False)

    def call(self, input):
        return input - self.const

