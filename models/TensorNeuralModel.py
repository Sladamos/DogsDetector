from models.NeuralModel import NeuralModel
from tensorflow import keras


class TensorNeuralModel(NeuralModel):
    def evaluate(self, test_set, test_labels, number_of_samples):
        return self.model.evaluate(test_set, test_labels, batch_size=number_of_samples, verbose=0)

    def __init__(self):
        self.model = keras.models.Sequential()

    def compile(self, optimizer, loss, metrics):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def save_to_file(self, file_name):
        self.model.save(file_name)

    def train(self, training_set, validation_data, epochs, number_of_samples, callbacks=None, verbose=2):
        if callbacks is None:
            self.model.fit(training_set, validation_data, batch_size=number_of_samples, epochs=epochs, verbose=verbose)
        else:
            self.model.fit(training_set, validation_data, batch_size=number_of_samples, epochs=epochs, verbose=verbose,
                           callbacks=callbacks)

    def add_layer(self, layer):
        self.model.add(layer)
