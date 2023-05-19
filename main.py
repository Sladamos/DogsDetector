import os
import sys

from PyQt5.QtWidgets import QApplication
from keras.preprocessing.image import ImageDataGenerator

from data.loaders.CifarDataLoader import CifarDataLoader
from data.loaders.DogsDataLoader import DogsDataLoader
from data.normalizators.DivideNormalizator import DivideNormalizator
from displayer.DogsImagesDisplayer import DogsImagesDisplayer
from gui.Application import Application
from models.comparators.ModelComparator import ModelComparator

import tensorflow as tf
import matplotlib.pyplot as plt
from models.creators.CifarModelsCreator import CifarModelsCreator
from models.creators.DogsModelsCreator import DogsModelsCreator
from models.loaders.TensorModelLoader import TensorModelLoader
from models.savers.TensorModelSaver import TensorModelSaver

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
    path = os.path.normpath("./images/dogs/All_images")
    dirs = os.listdir(path)
    models_creator = DogsModelsCreator(len(dirs))
    data_loader = DogsDataLoader(batch_size, path)
    cnn_model = models_creator.create_advanced_neural_model()
    train_data = data_loader.load_train_data()
    validation_data = data_loader.load_validation_data()
    epochs = 20
    verbose = 1

    checkpoint = tf.keras.callbacks.ModelCheckpoint("./newHope/transfered", monitor='val_accuracy', save_best_only=True)
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
    results = cnn_model.train_with_validation(train_data, validation_data, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[earlystop, checkpoint])
    saver.save_model(cnn_model, "./newHope/saved/transfered")
    make_plots(results, "transfered.png")


def init_app():
    data_loader = CifarDataLoader()
    normalizator = DivideNormalizator(255.0)
    model_loader = TensorModelLoader()
    return data_loader, model_loader, normalizator


def train_model():
    batch_size = 64
    path = os.path.normpath("./images/dogs/Images")
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
    checkpoint = tf.keras.callbacks.ModelCheckpoint("./newHope/simple", monitor='val_accuracy', save_best_only=True)
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15)

    results = cnn_model.train_with_validation(train_data, validation_data, epochs=epochs, batch_size=batch_size,
                                              verbose=verbose, callbacks=[earlystop, checkpoint])
    # results = cnn_model.train_with_validation(train_data, validation_data, epochs=epochs, batch_size=batch_size, verbose=verbose)
    saver.save_model(cnn_model, "./newHope/saved/simple")
    make_plots(results, "simple.png")


def print_images():
    displayer = DogsImagesDisplayer()
    displayer.display_images()
    displayer.display_transformed_image()


# print_images()
# workbench()
# train_model()
data_loader, model_loader, normalizator = init_app()
app = QApplication(sys.argv)
my_app = Application(data_loader, model_loader, normalizator)
sys.exit(app.exec_())
