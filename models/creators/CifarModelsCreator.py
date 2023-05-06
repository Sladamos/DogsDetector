from models.TensorNeuralModel import TensorNeuralModel
from layers.creators.TensorLayersCreator import TensorLayersCreator
from models.creators.ModelsCreator import ModelsCreator
from tensorflow import keras


class CifarModelsCreator(ModelsCreator):
    def __init__(self):
        self.layers_creator = TensorLayersCreator()

    def create_advanced_neural_model(self):
        model = TensorNeuralModel()
        layers_creator = self.layers_creator
        model.add_layer(layers_creator.create_convolution_layer(32, 3, 'relu', input_shape=(32, 32, 3)))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_convolution_layer(32, 3, 'relu'))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_flatten_layer())
        model.add_layer(layers_creator.create_dense_layer(64))
        model.add_layer(layers_creator.create_dense_layer(10))

        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optim = keras.optimizers.Adam(learning_rate=0.001)
        metrics = ["accuracy"]
        model.compile(optim, loss, metrics)
        return model

    def create_simple_neural_model(self):
        model = TensorNeuralModel()
        layers_creator = self.layers_creator
        model.add_layer(layers_creator.create_flatten_layer(input_shape=(32, 32, 3)))
        model.add_layer(layers_creator.create_dense_layer(128, activation='relu'))
        model.add_layer(layers_creator.create_dense_layer(10))

        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optim = keras.optimizers.Adam(learning_rate=0.001)
        metrics = ["accuracy"]
        model.compile(optim, loss, metrics)
        return model
