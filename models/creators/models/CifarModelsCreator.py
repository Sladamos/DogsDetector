from models.TensorNeuralModel import TensorNeuralModel
from models.creators.TensorLayersCreator import TensorLayersCreator
from models.creators.models.ModelsCreator import ModelsCreator
from tensorflow import keras


class CifarModelsCreator(ModelsCreator):
    def __init__(self):
        self.layers_creator = TensorLayersCreator()

    def create_convolution_neural_model(self):
        model = TensorNeuralModel()
        layers_creator = self.layers_creator
        model.add_layer(layers_creator.createConvolutionLayer(32, 3, 'relu', input_shape=(32, 32, 3)))
        model.add_layer(layers_creator.createPoolLayer((2, 2)))
        model.add_layer(layers_creator.createConvolutionLayer(32, 3, 'relu'))
        model.add_layer(layers_creator.createPoolLayer((2, 2)))
        model.add_layer(layers_creator.createFlattenLayer())
        model.add_layer(layers_creator.createDenseLayer(64))
        model.add_layer(layers_creator.createDenseLayer(10))

        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optim = keras.optimizers.Adam(lr=0.001)
        metrics = ["accuracy"]
        model.compile(optim, loss, metrics)
        return model

    def create_simple_neural_model(self):
        model = TensorNeuralModel()
        layers_creator = self.layers_creator
        model.add_layer(layers_creator.createFlattenLayer())
        keras.layers.Flatten(input_shape=(32, 32, 3)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10),

        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optim = keras.optimizers.Adam(lr=0.001)
        metrics = ["accuracy"]
        model.compile(optim, loss, metrics)
        return model