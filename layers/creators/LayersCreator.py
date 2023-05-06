from abc import abstractmethod


class LayersCreator:
    @abstractmethod
    def create_convolution_layer(self, filters, kernel_size, activation, strides, padding, input_shape):
        pass

    @abstractmethod
    def create_pool_layer(self, pool_size):
        pass

    @abstractmethod
    def create_flatten_layer(self, input_shape):
        pass

    @abstractmethod
    def create_dense_layer(self, units, activation):
        pass
