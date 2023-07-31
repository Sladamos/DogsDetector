import json
import os
import sys

from keras.preprocessing.image import ImageDataGenerator

from data.loaders.DogsDataLoader import DogsDataLoader
from data.normalizers.DivideNormalizer import DivideNormalizer
from detector.DogsDetectorsFactory import DogsDetectorsFactory

import tensorflow as tf
import matplotlib.pyplot as plt
from models.creators.DogsModelsCreator import DogsModelsCreator
from models.loaders.TensorModelLoader import TensorModelLoader
from models.savers.TensorModelSaver import TensorModelSaver
from option.app.QtAppOption import QtAppOption
from option.app.ConsoleAppOption import ConsoleAppOption
from option.train.TrainSimpleOption import TrainSimpleOption
from option.train.TrainTransferredOption import TrainTransferredOption

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def make_plots(results, name):
    plt.plot(results.history['accuracy'])
    plt.plot(results.history['val_accuracy'])
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['training', 'validation'], loc='upper left')
    plt.savefig('./newHope/accuracy/' + name)
    plt.show()

    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['training', 'validation'], loc='upper left')
    plt.savefig('./newHope/loss/' + name)
    plt.show()


def workbench():
    batch_size = 32
    saver = TensorModelSaver()
    path = os.path.normpath("./images/all_dogs")
    dirs = os.listdir(path)
    models_creator = DogsModelsCreator(len(dirs))
    data_loader = DogsDataLoader(batch_size, path)
    cnn_model = models_creator.create_advanced_neural_model()
    train_data = data_loader.load_train_data()
    epochs = 1
    verbose = 1

    checkpoint = tf.keras.callbacks.ModelCheckpoint("./newHope/transfered", monitor='val_accuracy', save_best_only=True)
    cnn_model.train(train_data, epochs, batch_size, verbose=verbose, callbacks=[checkpoint])
    saver.save_model(cnn_model, "./created_models/transfered")


def init_app():
    data_loader = DogsDataLoader()
    data_normalizator = DivideNormalizer(255.0)
    model_loader = TensorModelLoader()
    detector = DogsDetectorsFactory(model_loader).create_detector()
    return data_loader, data_normalizator, detector


def train_model():
    batch_size = 64
    path = os.path.normpath("./images/all_dogs")
    dirs = os.listdir(path)
    models_creator = DogsModelsCreator(len(dirs))
    data_loader = DogsDataLoader(batch_size, path)
    cnn_model = models_creator.create_simple_neural_model(input_shape=(224, 224, 3))
    # loader = TensorModelLoader()
    # cnn_model = loader.load_model("./newHope/saved/our")
    saver = TensorModelSaver()
    train_data = data_loader.load_train_data()
    validation_data = data_loader.load_validation_data()
    epochs = 150
    verbose = 1

    results = cnn_model.train_with_validation(train_data, validation_data, epochs=epochs, batch_size=batch_size,
                                              verbose=verbose, callbacks=[])
    make_plots(results, "simple.png")
    # trainer class, it should also save classes names in some binary file


def open_config(file_name):
    try:
        f = open(file_name)
        config = json.load(f)
        f.close()
    except:
        print("Problem with config.json")
        exit(404)
    return config


def main():
    options = {
        "console": [ConsoleAppOption, "app"],
        "qt": [QtAppOption, "app"]
        "transferred": [TrainTransferredOption, "console_app"],
        "simple": [TrainSimpleOption, "console_app"]
    }
    if len(sys.argv) != 2 or sys.argv[1] not in options:
        print("Please give one of program options:")
        [print(option) for option in options]
        return
    option_str = sys.argv[1]
    config = open_config("config.json")
    option, config_name = options[option_str]
    option_config = config[config_name]
    option = option()
    option.execute(option_config)


main()
