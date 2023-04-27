from data.Data import Data
from data.normalizators.Normalizator import Normalizator


class DivideNormalizator(Normalizator):
    def __init__(self, divider):
        self.divider = divider

    def normalize(self, data):
        images = data.get_images().copy()
        normalized = images / self.divider
        return Data(normalized, data.get_labels().copy())
