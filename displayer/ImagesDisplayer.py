from abc import abstractmethod


class ImagesDisplayer:
    @abstractmethod
    def display_transformed_image(self):
        pass

    @abstractmethod
    def display_images(self):
        pass
