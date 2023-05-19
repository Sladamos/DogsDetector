from abc import abstractmethod


class ImagesDisplayer:
    @abstractmethod
    def display_transformed_image(self):
        pass

    @abstractmethod
    def display_images(self):
        pass

    @abstractmethod
    def display_image_with_gaussian_noise(self, path, std_dev):
        pass
