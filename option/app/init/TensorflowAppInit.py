from data.loaders.DogsDataLoader import DogsDataLoader
from data.normalizators.DivideNormalizator import DivideNormalizator
from detector.DogsDetectorsFactory import DogsDetectorsFactory
from models.loaders.TensorModelLoader import TensorModelLoader
from option.Option import Option


class TensorflowAppInit(Option):
    def __init__(self):
        self.data_loaders = {
            "dog": lambda: DogsDataLoader
        }

        self.detectors_factories = {
            "dog": lambda x: DogsDetectorsFactory
        }

    def execute(self, config):
        #TODO config
        detected_type = "dog"
        data_loader = self.data_loaders[detected_type]()
        data_normalizer = DivideNormalizator(255.0)
        model_loader = TensorModelLoader()
        detectors_factory = self.detectors_factories[detected_type](model_loader)
        return data_loader, data_normalizer, detectors_factory