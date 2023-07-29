from data.loaders.DogsDataLoader import DogsDataLoader
from data.normalizers.DivideNormalizer import DivideNormalizer
from detector.DogsDetectorsFactory import DogsDetectorsFactory
from initializers.AppInitializer import AppInitializer
from models.loaders.TensorModelLoader import TensorModelLoader


class TensorflowAppInitializer(AppInitializer):
    def __init__(self, config):
        self.data_loaders = {
            "dog": lambda: DogsDataLoader
        }

        self.detectors_factories = {
            "dog": lambda x: DogsDetectorsFactory
        }
        self.config = config

    def initialize_app(self, detected_type):
        data_loader = self.data_loaders[detected_type]()
        data_normalizer = DivideNormalizer(255.0)
        model_loader = TensorModelLoader()
        detectors_factory = self.detectors_factories[detected_type](model_loader)
        return data_loader, data_normalizer, detectors_factory
