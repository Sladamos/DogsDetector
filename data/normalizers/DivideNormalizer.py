from data.Data import Data
from data.normalizers.Normalizer import Normalizer


class DivideNormalizer(Normalizer):
    def __init__(self, divider):
        self.divider = divider

    def normalize(self, data):
        images = data.get_images().copy()
        labels = data.get_labels()
        if labels is not None:
            labels = labels.copy()
        normalized = images / self.divider
        return Data(normalized, labels)
