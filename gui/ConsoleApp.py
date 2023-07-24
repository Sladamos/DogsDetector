import imghdr
import os

from gui.App import App


class ConsoleApp(App):

    def __init__(self, data_loader, data_normalizator, detector):
        super().__init__(data_loader, data_normalizator, detector)
        self.image_path = ""
        self.app_options = {
            "Select image path": self.select_image_path,
            "Load image": self.__load_img,
            "Get classification result": self.get_classification_result,
            "Exit": self.disable_app
        }
        self.indexed_options = [key for key in self.app_options]
        self.is_program_launched = False

    def select_image_path(self):
        self.forget_image()
        image_path = input('Select image path: ')
        image_path = os.path.normpath(image_path)
        if not os.path.isfile(image_path) or imghdr.what(image_path) is None:
            image_path = ""
            print("Incorrect image path")
        self.image_path = image_path
        print('Selected path: ' + image_path)

    def show(self):
        self.is_program_launched = True

        while self.is_program_launched:
            self.__print_options()
            option = input('Select option: ')
            if option in self.app_options:
                self.app_options[option]()
            elif option.isnumeric() and int(option)-1 < len(self.indexed_options):
                self.app_options[self.indexed_options[int(option)-1]]()
            else:
                print("Incorrect option")
        pass

    def disable_app(self):
        self.is_program_launched = False

    def __load_img(self):
        if self.image_path != "":
            self.load_image(self.image_path)
            print("Image loaded properly")
        else:
            print("Incorrect image path")

    def __print_options(self):
        for i in range(len(self.indexed_options)):
            print("%d: %s" % (i+1, self.indexed_options[i]))


    def get_classification_result(self):
        if self.is_image_selected():
            return super().get_classification_result()
        else:
            print("Select image first")

