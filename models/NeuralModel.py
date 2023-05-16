from abc import abstractmethod


class NeuralModel:
    @abstractmethod
    def disableLayer(self, layer_number):
        pass

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
    def train(self, train_data, epochs, batch_size, callbacks, verbose):
        pass

    @abstractmethod
    def train_with_validation(self, train_data, validation_data, epochs, batch_size, callbacks, verbose):
        pass

    @abstractmethod
    def evaluate(self, test_data, number_of_samples):
        pass

    @abstractmethod
    def predict(self, data):
        pass

