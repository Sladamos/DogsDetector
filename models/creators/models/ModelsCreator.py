from abc import abstractmethod


class ModelsCreator:

    @abstractmethod
    def create_simple_neural_model(self):
        pass

    @abstractmethod
    def create_convolution_neural_model(self):
        pass