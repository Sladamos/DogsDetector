from layers.creators.TensorLayersCreator import TensorLayersCreator
from models.TensorNeuralModel import TensorNeuralModel
from models.creators.ModelsCreator import ModelsCreator
from tensorflow import keras


class DogsModelsCreator(ModelsCreator):
    def __init__(self):
        self.layers_creator = TensorLayersCreator()
        self.number_of_breeds = 120

    def create_advanced_neural_model(self):
        model = TensorNeuralModel()
        layers_creator = self.layers_creator
        model.add_layer(layers_creator.create_convolution_layer(32, 3, 'relu', input_shape=(224, 224, 3), padding='same'))
        model.add_layer(layers_creator.create_batch_normalization_layer())
        model.add_layer(layers_creator.create_convolution_layer(32, 3, 'relu', padding='same'))
        model.add_layer(layers_creator.create_batch_normalization_layer())
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_dropout_layer(0.25))

        model.add_layer(layers_creator.create_convolution_layer(64, 3, 'relu', padding='same'))
        model.add_layer(layers_creator.create_batch_normalization_layer())
        model.add_layer(layers_creator.create_convolution_layer(64, 3, 'relu', padding='same'))
        model.add_layer(layers_creator.create_batch_normalization_layer())
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_dropout_layer(0.25))

        model.add_layer(layers_creator.create_convolution_layer(128, 3, 'relu', padding='same'))
        model.add_layer(layers_creator.create_batch_normalization_layer())
        model.add_layer(layers_creator.create_convolution_layer(128, 3, 'relu', padding='same'))
        model.add_layer(layers_creator.create_batch_normalization_layer())
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_dropout_layer(0.25))

        model.add_layer(layers_creator.create_convolution_layer(256, 3, 'relu', padding='same'))
        model.add_layer(layers_creator.create_batch_normalization_layer())
        model.add_layer(layers_creator.create_convolution_layer(256, 3, 'relu', padding='same'))
        model.add_layer(layers_creator.create_batch_normalization_layer())
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_dropout_layer(0.25))

        model.add_layer(layers_creator.create_flatten_layer())
        model.add_layer(layers_creator.create_dense_layer(512, activation='relu'))
        model.add_layer(layers_creator.create_batch_normalization_layer())
        model.add_layer(layers_creator.create_dropout_layer(0.5))
        model.add_layer(layers_creator.create_dense_layer(self.number_of_breeds, activation='softmax'))

        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optim = keras.optimizers.Adam(learning_rate=0.001)
        metrics = ["accuracy"]
        model.compile(optim, loss, metrics)
        return model

    def create_simple_neural_model(self):
        model = TensorNeuralModel()
        layers_creator = self.layers_creator
        model.add_layer(layers_creator.create_convolution_layer(32, 3, 'relu', input_shape=(224, 224, 3)))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_batch_normalization_layer())
        model.add_layer(layers_creator.create_convolution_layer(64, 3, 'relu'))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_batch_normalization_layer())
        model.add_layer(layers_creator.create_convolution_layer(128, 3, 'relu'))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_batch_normalization_layer())
        model.add_layer(layers_creator.create_convolution_layer(256, 3, 'relu'))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_batch_normalization_layer())
        model.add_layer(layers_creator.create_flatten_layer())
        model.add_layer(layers_creator.create_dense_layer(512, activation='relu'))
        model.add_layer(layers_creator.create_dropout_layer(0.5))
        model.add_layer(layers_creator.create_dense_layer(256, activation='relu'))
        model.add_layer(layers_creator.create_dropout_layer(0.5))
        model.add_layer(layers_creator.create_dense_layer(self.number_of_breeds, activation='softmax'))

        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optim = keras.optimizers.Adam(learning_rate=0.001)
        metrics = ["accuracy"]
        model.compile(optim, loss, metrics)
        return model
