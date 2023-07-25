from detector.Detector import Detector
from detector.DetectorsFactory import DetectorsFactory


class DogsDetectorsFactory(DetectorsFactory):
    def __init__(self, model_loader):
        self.model_loader = model_loader
        self.models_paths = {
            "simple": "./created_models/simple_model",
            "transfered": "./created_models/transfered_model"
        }

        self.images_paths = {
            "simple": "./images/simple_dogs",
            "transfered": "./images/all_dogs"
        }

    def create_detector(self):
        return Detector(self.model_loader, self.models_paths, self.images_paths)
