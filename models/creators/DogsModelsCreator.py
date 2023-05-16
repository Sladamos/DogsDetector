from layers.creators.TensorLayersCreator import TensorLayersCreator
from models.TensorNeuralModel import TensorNeuralModel
from models.creators.ModelsCreator import ModelsCreator
from tensorflow import keras


class DogsModelsCreator(ModelsCreator):
    def __init__(self, number_of_classes):
        self.layers_creator = TensorLayersCreator()
        self.number_of_breeds = number_of_classes

    def create_advanced_neural_model(self, input_shape):
        model = TensorNeuralModel()
        layers_creator = self.layers_creator
        model.add_layer(layers_creator.create_convolution_layer(32, 3, 'relu', input_shape=input_shape))
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

        model.compile('adam', 'categorical_crossentropy', ['accuracy'])

        return model

    def create_simple_neural_model(self, input_shape=(224, 224, 3)):
        model = self.model_6(input_shape)
        loss = keras.losses.CategoricalCrossentropy()
        optim = keras.optimizers.Adam(learning_rate=0.001)
        metrics = ["accuracy"]
        model.compile(optim, loss, metrics)
        return model

    def model_1(self, input_shape):
        model = TensorNeuralModel()
        layers_creator = self.layers_creator
        model.add_layer(layers_creator.create_flatten_layer(input_shape=input_shape))
        model.add_layer(layers_creator.create_dense_layer(self.number_of_breeds, activation='softmax'))
        return model

    # def model_abandoned_1(self, input_shape):
        # model = TensorNeuralModel()
        # layers_creator = self.layers_creator
        # model.add_layer(layers_creator.create_flatten_layer(input_shape=input_shape))
        # model.add_layer(layers_creator.create_dense_layer(2*self.number_of_breeds, activation='relu'))
        # model.add_layer(layers_creator.create_dense_layer(self.number_of_breeds, activation='softmax'))
        # return model

    def model_2(self, input_shape):
        model = TensorNeuralModel()
        layers_creator = self.layers_creator
        model.add_layer(layers_creator.create_convolution_layer(32, 3, 'relu', input_shape=input_shape))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_flatten_layer(input_shape=input_shape))
        model.add_layer(layers_creator.create_dense_layer(self.number_of_breeds, activation='softmax'))
        return model

    def model_3(self, input_shape):
        model = TensorNeuralModel()
        layers_creator = self.layers_creator
        model.add_layer(layers_creator.create_convolution_layer(32, 3, 'relu', input_shape=input_shape))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_dropout_layer(0.15))
        model.add_layer(layers_creator.create_flatten_layer(input_shape=input_shape))
        model.add_layer(layers_creator.create_dense_layer(self.number_of_breeds, activation='softmax'))
        return model

    # def model_abandoned_2(self, input_shape):
    #     model = TensorNeuralModel()
    #     layers_creator = self.layers_creator
    #     model.add_layer(layers_creator.create_convolution_layer(32, 3, 'relu', input_shape=input_shape))
    #     model.add_layer(layers_creator.create_gaussian_noise(0.01))
    #     model.add_layer(layers_creator.create_pool_layer((2, 2)))
    #     model.add_layer(layers_creator.create_dropout_layer(0.15))
    #
    #     model.add_layer(layers_creator.create_flatten_layer(input_shape=input_shape))
    #     model.add_layer(layers_creator.create_dense_layer(self.number_of_breeds, activation='softmax'))
    #     return model

    def model_4(self, input_shape):
        model = TensorNeuralModel()
        layers_creator = self.layers_creator
        model.add_layer(layers_creator.create_convolution_layer(32, 3, 'relu', input_shape=input_shape))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_dropout_layer(0.15))

        model.add_layer(layers_creator.create_convolution_layer(64, 3, 'relu', input_shape=input_shape))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_dropout_layer(0.15))

        model.add_layer(layers_creator.create_flatten_layer(input_shape=input_shape))
        model.add_layer(layers_creator.create_dense_layer(self.number_of_breeds, activation='softmax'))
        return model

    def model_5(self, input_shape):
        model = TensorNeuralModel()
        layers_creator = self.layers_creator
        model.add_layer(layers_creator.create_convolution_layer(32, 3, 'relu', input_shape=input_shape))
        model.add_layer(layers_creator.create_gaussian_noise(0.01))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_dropout_layer(0.15))

        model.add_layer(layers_creator.create_convolution_layer(64, 3, 'relu', input_shape=input_shape))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_dropout_layer(0.15))

        model.add_layer(layers_creator.create_convolution_layer(128, 3, 'relu', input_shape=input_shape))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_dropout_layer(0.15))

        model.add_layer(layers_creator.create_flatten_layer(input_shape=input_shape))
        model.add_layer(layers_creator.create_dropout_layer(0.25))
        model.add_layer(layers_creator.create_dense_layer(self.number_of_breeds, activation='softmax'))
        return model

    def model_6(self, input_shape):
        model = TensorNeuralModel()
        layers_creator = self.layers_creator
        model.add_layer(layers_creator.create_convolution_layer(32, 3, 'relu', input_shape=input_shape))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_batch_normalization_layer())
        model.add_layer(layers_creator.create_dropout_layer(0.25))

        model.add_layer(layers_creator.create_convolution_layer(64, 3, 'relu', input_shape=input_shape))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_batch_normalization_layer())
        model.add_layer(layers_creator.create_dropout_layer(0.25))
        model.add_layer(layers_creator.create_gaussian_noise(0.0025))

        model.add_layer(layers_creator.create_convolution_layer(128, 3, 'relu', input_shape=input_shape))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_batch_normalization_layer())
        model.add_layer(layers_creator.create_dropout_layer(0.25))
        model.add_layer(layers_creator.create_gaussian_noise(0.0025))

        model.add_layer(layers_creator.create_convolution_layer(128, 3, 'relu', input_shape=input_shape))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_batch_normalization_layer())
        model.add_layer(layers_creator.create_dropout_layer(0.25))

        model.add_layer(layers_creator.create_convolution_layer(256, 3, 'relu', input_shape=input_shape))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_batch_normalization_layer())
        model.add_layer(layers_creator.create_dropout_layer(0.25))

        model.add_layer(layers_creator.create_flatten_layer(input_shape=input_shape))
        model.add_layer(layers_creator.create_dense_layer(512, activation='relu'))
        model.add_layer(layers_creator.create_dropout_layer(0.25))
        model.add_layer(layers_creator.create_dense_layer(256, activation='relu'))
        model.add_layer(layers_creator.create_dropout_layer(0.25))
        model.add_layer(layers_creator.create_dense_layer(128, activation='relu'))
        model.add_layer(layers_creator.create_dropout_layer(0.25))
        model.add_layer(layers_creator.create_dense_layer(self.number_of_breeds, activation='softmax'))
        return model