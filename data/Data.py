from abc import abstractmethod


class Data:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def get_images(self):
        return self.images

    def get_labels(self):
        return self.labels
