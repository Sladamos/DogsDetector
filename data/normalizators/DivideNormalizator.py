from data.Data import Data
from data.normalizators.Normalizator import Normalizator


class DivideNormalizator(Normalizator):
    def __init__(self, divider):
        self.divider = divider

    def normalize(self, data):
        images = data.get_images().copy()
        labels = data.get_labels()
        if labels is not None:
            labels = labels.copy()
        images = images / self.divider
        return Data(images, labels)
