from models.TensorNeuralModel import TensorNeuralModel
from layers.creators.TensorLayersCreator import TensorLayersCreator
from models.creators.ModelsCreator import ModelsCreator
from tensorflow import keras


class CifarModelsCreator(ModelsCreator):
    def create_simple_neural_model(self, input_shape):
        pass

    def __init__(self, number_of_classes=10):
        self.layers_creator = TensorLayersCreator()
        self.number_of_classes = number_of_classes

    def create_advanced_neural_model(self, input_shape=(32, 32, 3)):
        model = TensorNeuralModel()
        layers_creator = self.layers_creator
        model.add_layer(layers_creator.create_convolution_layer(32, 3, 'relu', input_shape=input_shape))
        model.add_layer(layers_creator.create_batch_normalization_layer())
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_dropout_layer(0.25))

        model.add_layer(layers_creator.create_convolution_layer(64, 3, 'relu'))
        model.add_layer(layers_creator.create_batch_normalization_layer())
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_dropout_layer(0.25))

        model.add_layer(layers_creator.create_convolution_layer(128, 3, 'relu'))
        model.add_layer(layers_creator.create_batch_normalization_layer())
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_dropout_layer(0.25))

        model.add_layer(layers_creator.create_convolution_layer(256, 3, 'relu'))
        model.add_layer(layers_creator.create_batch_normalization_layer())
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_dropout_layer(0.25))

        model.add_layer(layers_creator.create_flatten_layer())
        model.add_layer(layers_creator.create_dense_layer(1024, activation='relu'))
        model.add_layer(layers_creator.create_batch_normalization_layer())
        model.add_layer(layers_creator.create_dropout_layer(0.5))
        model.add_layer(layers_creator.create_dense_layer(512, activation='relu'))
        model.add_layer(layers_creator.create_batch_normalization_layer())
        model.add_layer(layers_creator.create_dropout_layer(0.5))
        model.add_layer(layers_creator.create_dense_layer(self.number_of_classes, activation='softmax'))

        loss = keras.losses.CategoricalCrossentropy()
        # optim = keras.optimizers.SGD(learning_rate=0.001)
        optim = keras.optimizers.Adam(learning_rate=0.001)
        metrics = ["accuracy"]
        model.compile(optim, loss, metrics)
        return model