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
    def train(self, train_data, epochs, number_of_samples, callbacks):
        pass

    @abstractmethod
    def evaluate(self, test_data, number_of_samples):
        pass

    # TODO: implement predict

