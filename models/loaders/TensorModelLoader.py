from models.TensorNeuralModel import TensorNeuralModel
from models.loaders.ModelLoader import ModelLoader
import tensorflow as tf


class TensorModelLoader(ModelLoader):
    def load_model(self, path):
        model = tf.keras.models.load_model(path)
        tensor = TensorNeuralModel()
        tensor.set_model(model)
        return tensor
