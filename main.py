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
        "qt": [QtAppOption, "app"],
        "transferred": [TrainTransferredOption, "train"],
        "simple": [TrainSimpleOption, "train"]
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
