from abc import abstractmethod


class NeuralModel:

    @abstractmethod
    def add_layer(self, layer):
        pass

    @abstractmethod
    def compile(self, optimizer, loss, metrics):
        pass

    @abstractmethod
    def save_to_file(self, file_name):
        pass

    @abstractmethod
    def train(self, training_set, validation_data, epochs, number_of_samples, callbacks):
        pass

    @abstractmethod
    def evaluate(self, test_set, test_labels, number_of_samples):
        pass

