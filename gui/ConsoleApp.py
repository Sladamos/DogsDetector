import imghdr
import os

from gui.App import App


class ConsoleApp(App):

    def __init__(self, data_loader, data_normalizator, detector):
        super().__init__(data_loader, data_normalizator, detector)
        self.image_path = None
        self.app_options = {
            "select_image_path": self.select_image_path,
            "load_image": self.__load_img,
            "get_classification_result": self.get_classification_result,
            "exit": self.disable_app
        }
        self.is_program_launched = False

    def select_image_path(self):
        image_path = input('Select image path')
        if not os.path.isfile(image_path) or imghdr.what(image_path) is None:
            image_path = ""
        self.image_path = image_path

    def show(self):
        self.is_program_launched = True
        while self.is_program_launched:
            option = input('Select option')
            if option in self.app_options:
                self.app_options[option]()
            else:
                print("Incorrect option")
        pass

    def disable_app(self):
        self.is_program_launched = False

    def __load_img(self):
        if self.image_path != "":
            self.load_image(self.image_path)
        else
            print("Incorrect image path")

