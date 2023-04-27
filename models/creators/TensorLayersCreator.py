from models.creators.LayersCreator import LayersCreator
from tensorflow import keras
from keras import layers


class TensorLayersCreator(LayersCreator):
    def createConvolutionLayer(self, filters, kernel_size, activation=None, strides=(1, 1), padding="valid",
                               input_shape=None):
        if input_shape is not None:
            return layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, activation=activation,
                                 input_shape=input_shape)
        else:
            return layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, activation=activation)

    def createPoolLayer(self, pool_size):
        return layers.MaxPool2D(pool_size)

    def createFlattenLayer(self, input_shape=None):
        if input_shape is None:
            return layers.Flatten()
        else:
            return layers.Flatten(input_shape=input_shape)

    def createDenseLayer(self, units, activation=None):
        return layers.Dense(units, activation=activation)
