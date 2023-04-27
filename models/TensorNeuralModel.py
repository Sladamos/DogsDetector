import numpy as np

from data.Data import Data
from models.NeuralModel import NeuralModel
from tensorflow import keras


class TensorNeuralModel(NeuralModel):

    def predict(self, data):
        images = data.get_images()
        predicted_images = self.model.predict(images, verbose=0)
        formatted_images = np.argmax(predicted_images, axis=1)
        return formatted_images

    def evaluate(self, test_data, number_of_samples):
        test_set = test_data.get_images()
        test_labels = test_data.get_labels()
        return self.model.evaluate(test_set, test_labels, batch_size=number_of_samples, verbose=0)

    def __init__(self):
        self.model = keras.models.Sequential()

    def compile(self, optimizer, loss, metrics):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def save_to_file(self, file_name):
        self.model.save(file_name)

    def train(self, training_data, epochs, number_of_samples, callbacks=None, verbose=2):
        training_set = training_data.get_images()
        training_labels = training_data.get_labels()
        if callbacks is None:
            self.model.fit(training_set, training_labels, batch_size=number_of_samples, epochs=epochs, verbose=verbose)
        else:
            self.model.fit(training_set, training_labels, batch_size=number_of_samples, epochs=epochs, verbose=verbose,
                           callbacks=callbacks)

    def add_layer(self, layer):
        self.model.add(layer)
