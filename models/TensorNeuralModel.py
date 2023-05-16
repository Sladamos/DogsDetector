import numpy as np

from data.Data import Data
from models.NeuralModel import NeuralModel
from tensorflow import keras


class TensorNeuralModel(NeuralModel):
    def disableLayer(self, layer_number):
        self.model.layers[layer_number].trainable = False

    def train_with_validation(self, train_data, validation_data, epochs, batch_size, callbacks=None, verbose=2):
        training_set = train_data.get_images()
        validation_set = validation_data.get_images()
        if callbacks is None:
            return self.model.fit(training_set, validation_data=validation_set, batch_size=batch_size, epochs=epochs, verbose=verbose)
        else:
            return self.model.fit(training_set, validation_data=validation_set, batch_size=batch_size, epochs=epochs, verbose=verbose,
                           callbacks=callbacks)

    def __init__(self, base_model=None):
        self.model = keras.models.Sequential()
        if base_model is not None:
            self.model.add(base_model)

    def predict(self, data):
        images = data.get_images()
        predicted_images = self.model.predict(images, verbose=0)
        formatted_images = np.argmax(predicted_images, axis=1)
        return formatted_images

    def evaluate(self, test_data, number_of_samples):
        test_set = test_data.get_images()
        test_labels = test_data.get_labels()
        return self.model.evaluate(test_set, test_labels, batch_size=number_of_samples, verbose=0)

    def set_model(self, model):
        self.model = model

    def compile(self, optimizer, loss, metrics):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def save_to_file(self, file_name):
        self.model.save(file_name)

    def train(self, training_data, epochs, batch_size, callbacks=None, verbose=2):
        training_set = training_data.get_images()
        training_labels = training_data.get_labels()
        if callbacks is None:
            return self.model.fit(training_set, training_labels, batch_size=batch_size, epochs=epochs, verbose=verbose)
        else:
            return self.model.fit(training_set, training_labels, batch_size=batch_size, epochs=epochs, verbose=verbose,
                           callbacks=callbacks)

    def add_layer(self, layer):
        self.model.add(layer)
