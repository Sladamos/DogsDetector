from data.loaders.DogsDataLoader import DogsDataLoader
from data.normalizers.DivideNormalizer import DivideNormalizer
from initializers.trainer.TrainerInitializer import TrainerInitializer
from models.creators.DogsModelsCreator import DogsModelsCreator
from models.loaders.TensorModelLoader import TensorModelLoader
from models.savers.TensorModelSaver import TensorModelSaver


class TensorflowTrainerInitializer(TrainerInitializer):
    def __init__(self):
        self.data_loaders = {
            "dog": DogsDataLoader
        }

        self.models_creators = {
            "dog": lambda x: DogsModelsCreator(x)
        }

    def initialize_trainer(self, initializer_config):
        detected_type = initializer_config["detected_type"]
        data_loader = self.data_loaders[detected_type]()
        data_normalizer = DivideNormalizer(255.0)
        train_from_existing = initializer_config["train_from_existing"]
        if train_from_existing:
            model_operator = TensorModelLoader()
        else:
            model_operator = self.models_creators[detected_type](initializer_config["number_of_classes"])
        model_saver = TensorModelSaver()
        return data_loader, data_normalizer, model_operator, model_saver
