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

    @abstractmethod
    def create_batch_normalization_layer(self):
        pass

    @abstractmethod
    def create_dropout_layer(self, rate):
        pass

    @abstractmethod
    def create_gaussian_noise(self, standard_deviation):
        pass