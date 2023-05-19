from layers.creators.TensorLayersCreator import TensorLayersCreator
from models.TensorNeuralModel import TensorNeuralModel
from models.creators.ModelsCreator import ModelsCreator
from tensorflow import keras
from keras.applications.inception_v3 import InceptionV3


class DogsModelsCreator(ModelsCreator):
    def __init__(self, number_of_classes):
        self.layers_creator = TensorLayersCreator()
        self.number_of_breeds = number_of_classes

    def create_advanced_neural_model(self, input_shape=(224, 224, 3)):
        model = self.adv_model_2(input_shape=input_shape)

        model.compile(keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def adv_model_1(self, input_shape):
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
        model = TensorNeuralModel(base_model)
        model.add_layer(self.layers_creator.create_global_average_pooling())
        model.add_layer(self.layers_creator.create_dropout_layer(0.3))
        model.add_layer(self.layers_creator.create_dense_layer(512, activation="relu"))
        model.add_layer(self.layers_creator.create_dense_layer(512, activation="relu"))
        model.add_layer(self.layers_creator.create_dense_layer(self.number_of_breeds, activation='softmax'))
        model.disableLayer(0)
        return model

    def adv_model_2(self, input_shape):
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
        model = TensorNeuralModel(base_model)
        model.add_layer(self.layers_creator.create_global_average_pooling())
        model.add_layer(self.layers_creator.create_dropout_layer(0.2))
        model.add_layer(self.layers_creator.create_dense_layer(self.number_of_breeds, activation='softmax'))
        model.disableLayer(0)
        return model

    def create_simple_neural_model(self, input_shape=(224, 224, 3)):
        model = self.model_11(input_shape)
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
        model.add_layer(layers_creator.create_gaussian_noise(0.0025))
        model.add_layer(layers_creator.create_convolution_layer(32, 3, 'relu', input_shape=input_shape))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_flatten_layer(input_shape=input_shape))
        model.add_layer(layers_creator.create_dense_layer(self.number_of_breeds, activation='softmax'))
        return model

    def model_3(self, input_shape):
        model = TensorNeuralModel()
        layers_creator = self.layers_creator
        model.add_layer(layers_creator.create_gaussian_noise(0.0025))
        model.add_layer(layers_creator.create_convolution_layer(32, 3, 'relu', input_shape=input_shape))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_gaussian_noise(0.0025))
        model.add_layer(layers_creator.create_convolution_layer(64, 3, 'relu', input_shape=input_shape))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_flatten_layer(input_shape=input_shape))
        #model.add_layer(layers_creator.create_dense_layer(128, activation='relu'))
        #model.add_layer(layers_creator.create_dropout_layer(0.25))
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
        model.add_layer(layers_creator.create_gaussian_noise(0.0025))
        model.add_layer(layers_creator.create_convolution_layer(32, 3, 'relu', input_shape=input_shape))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_flatten_layer(input_shape=input_shape))
        model.add_layer(layers_creator.create_dense_layer(64, activation='relu'))
        model.add_layer(layers_creator.create_dropout_layer(0.25))
        model.add_layer(layers_creator.create_dense_layer(self.number_of_breeds, activation='softmax'))
        return model

    def model_5(self, input_shape):
        model = TensorNeuralModel()
        layers_creator = self.layers_creator
        model.add_layer(layers_creator.create_gaussian_noise(0.0025))
        model.add_layer(layers_creator.create_convolution_layer(32, 3, 'relu', input_shape=input_shape))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_dropout_layer(0.1))

        model.add_layer(layers_creator.create_gaussian_noise(0.0025))
        model.add_layer(layers_creator.create_convolution_layer(64, 3, 'relu', input_shape=input_shape))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_dropout_layer(0.1))

        model.add_layer(layers_creator.create_flatten_layer(input_shape=input_shape))
        model.add_layer(layers_creator.create_dense_layer(128, activation='relu'))
        model.add_layer(layers_creator.create_dropout_layer(0.25))
        model.add_layer(layers_creator.create_dense_layer(self.number_of_breeds, activation='softmax'))
        return model

    def model_6(self, input_shape):
        model = TensorNeuralModel()
        layers_creator = self.layers_creator
        model.add_layer(layers_creator.create_gaussian_noise(0.004))
        model.add_layer(layers_creator.create_convolution_layer(32, 3, 'relu', input_shape=input_shape))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_dropout_layer(0.25))

        model.add_layer(layers_creator.create_gaussian_noise(0.004))
        model.add_layer(layers_creator.create_convolution_layer(64, 3, 'relu', input_shape=input_shape))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_dropout_layer(0.25))

        model.add_layer(layers_creator.create_flatten_layer(input_shape=input_shape))
        model.add_layer(layers_creator.create_dense_layer(128, activation='relu'))
        model.add_layer(layers_creator.create_dropout_layer(0.4))
        model.add_layer(layers_creator.create_dense_layer(self.number_of_breeds, activation='softmax'))
        return model

    def model_7(self, input_shape):
        model = TensorNeuralModel()
        layers_creator = self.layers_creator
        model.add_layer(layers_creator.create_gaussian_noise(0.004))
        model.add_layer(layers_creator.create_convolution_layer(32, 3, 'relu', input_shape=input_shape))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_dropout_layer(0.25))

        model.add_layer(layers_creator.create_gaussian_noise(0.004))
        model.add_layer(layers_creator.create_convolution_layer(64, 3, 'relu', input_shape=input_shape))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_dropout_layer(0.25))

        model.add_layer(layers_creator.create_gaussian_noise(0.004))
        model.add_layer(layers_creator.create_convolution_layer(128, 3, 'relu', input_shape=input_shape))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_dropout_layer(0.25))

        model.add_layer(layers_creator.create_flatten_layer(input_shape=input_shape))
        model.add_layer(layers_creator.create_dense_layer(128, activation='relu'))
        model.add_layer(layers_creator.create_dropout_layer(0.4))
        model.add_layer(layers_creator.create_dense_layer(self.number_of_breeds, activation='softmax'))
        return model

    def model_8(self, input_shape):
        model = TensorNeuralModel()
        layers_creator = self.layers_creator
        model.add_layer(layers_creator.create_gaussian_noise(0.004))
        model.add_layer(layers_creator.create_convolution_layer(32, 3, 'relu', input_shape=input_shape))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_dropout_layer(0.25))

        model.add_layer(layers_creator.create_gaussian_noise(0.004))
        model.add_layer(layers_creator.create_convolution_layer(64, 3, 'relu', input_shape=input_shape))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_dropout_layer(0.25))

        model.add_layer(layers_creator.create_gaussian_noise(0.004))
        model.add_layer(layers_creator.create_convolution_layer(128, 3, 'relu', input_shape=input_shape))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_dropout_layer(0.25))

        model.add_layer(layers_creator.create_flatten_layer(input_shape=input_shape))
        model.add_layer(layers_creator.create_dense_layer(256, activation='relu'))
        model.add_layer(layers_creator.create_dropout_layer(0.4))
        model.add_layer(layers_creator.create_dense_layer(128, activation='relu'))
        model.add_layer(layers_creator.create_dropout_layer(0.4))
        model.add_layer(layers_creator.create_dense_layer(self.number_of_breeds, activation='softmax'))
        return model

    def model_9(self, input_shape):
        model = TensorNeuralModel()
        layers_creator = self.layers_creator
        model.add_layer(layers_creator.create_gaussian_noise(0.004))
        model.add_layer(layers_creator.create_convolution_layer(32, 3, 'relu', input_shape=input_shape))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_dropout_layer(0.25))

        model.add_layer(layers_creator.create_gaussian_noise(0.004))
        model.add_layer(layers_creator.create_convolution_layer(64, 3, 'relu', input_shape=input_shape))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_dropout_layer(0.25))

        model.add_layer(layers_creator.create_gaussian_noise(0.004))
        model.add_layer(layers_creator.create_convolution_layer(128, 3, 'relu', input_shape=input_shape))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_dropout_layer(0.25))

        model.add_layer(layers_creator.create_gaussian_noise(0.004))
        model.add_layer(layers_creator.create_convolution_layer(256, 3, 'relu', input_shape=input_shape))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_dropout_layer(0.25))

        model.add_layer(layers_creator.create_flatten_layer())
        model.add_layer(layers_creator.create_dense_layer(256, activation='relu'))
        model.add_layer(layers_creator.create_dropout_layer(0.45))
        model.add_layer(layers_creator.create_dense_layer(128, activation='relu'))
        model.add_layer(layers_creator.create_dropout_layer(0.45))
        model.add_layer(layers_creator.create_dense_layer(self.number_of_breeds, activation='softmax'))
        return model

    def model_10(self, input_shape):
        model = TensorNeuralModel()
        layers_creator = self.layers_creator
        model.add_layer(layers_creator.create_gaussian_noise(0.004))
        model.add_layer(layers_creator.create_convolution_layer(32, 3, 'relu', input_shape=input_shape))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_dropout_layer(0.25))

        model.add_layer(layers_creator.create_gaussian_noise(0.003))
        model.add_layer(layers_creator.create_convolution_layer(64, 3, 'relu', input_shape=input_shape))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_dropout_layer(0.25))

        model.add_layer(layers_creator.create_gaussian_noise(0.003))
        model.add_layer(layers_creator.create_convolution_layer(128, 3, 'relu', input_shape=input_shape))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_dropout_layer(0.25))

        model.add_layer(layers_creator.create_gaussian_noise(0.004))
        model.add_layer(layers_creator.create_convolution_layer(256, 3, 'relu', input_shape=input_shape))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
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

    def model_11(self, input_shape):
        model = TensorNeuralModel()
        layers_creator = self.layers_creator
        model.add_layer(layers_creator.create_gaussian_noise(0.004))
        model.add_layer(layers_creator.create_convolution_layer(32, 3, 'relu', input_shape=input_shape))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_dropout_layer(0.25))

        model.add_layer(layers_creator.create_gaussian_noise(0.004))
        model.add_layer(layers_creator.create_convolution_layer(64, 3, 'relu', input_shape=input_shape))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_dropout_layer(0.25))

        model.add_layer(layers_creator.create_gaussian_noise(0.004))
        model.add_layer(layers_creator.create_convolution_layer(128, 3, 'relu', input_shape=input_shape))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_dropout_layer(0.25))

        model.add_layer(layers_creator.create_gaussian_noise(0.004))
        model.add_layer(layers_creator.create_convolution_layer(256, 3, 'relu', input_shape=input_shape))
        model.add_layer(layers_creator.create_pool_layer((2, 2)))
        model.add_layer(layers_creator.create_dropout_layer(0.25))

        model.add_layer(layers_creator.create_flatten_layer(input_shape=input_shape))
        model.add_layer(layers_creator.create_dense_layer(1024, activation='relu'))
        model.add_layer(layers_creator.create_dropout_layer(0.5))
        model.add_layer(layers_creator.create_dense_layer(256, activation='relu'))
        model.add_layer(layers_creator.create_dropout_layer(0.5))
        model.add_layer(layers_creator.create_dense_layer(128, activation='relu'))
        model.add_layer(layers_creator.create_dropout_layer(0.5))
        model.add_layer(layers_creator.create_dense_layer(self.number_of_breeds, activation='softmax'))
        return model