from abc import abstractmethod


class ModelsCreator:
    @abstractmethod
    def create_simple_neural_model(self, number_of_classes):
        pass

    @abstractmethod
    def create_advanced_neural_model(self, number_of_classes):
        pass