from detector.Detector import Detector
from detector.DetectorsFactory import DetectorsFactory


class DogsDetectorsFactory(DetectorsFactory):
    def __init__(self, model_loader, models_configs):
        self.model_loader = model_loader
        self.models_configs = models_configs

    def create_detector(self):
        return Detector(self.model_loader, self.models_configs)
