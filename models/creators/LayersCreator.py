from abc import abstractmethod


class LayersCreator:
    @abstractmethod
    def createConvolutionLayer(self, filters, kernel_size,  activation, strides, padding, input_shape):
        pass

    @abstractmethod
    def createPoolLayer(self, pool_size):
        pass

    @abstractmethod
    def createFlattenLayer(self, input_shape):
        pass

    @abstractmethod
    def createDenseLayer(self, units, activation):
        pass
