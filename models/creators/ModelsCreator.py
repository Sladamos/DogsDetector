from abc import abstractmethod


class ModelsCreator:
    @abstractmethod
    def create_simple_neural_model(self, input_shape):
        pass

    @abstractmethod
    def create_advanced_neural_model(self, input_shape):
        pass