import os

from PyQt5.QtWidgets import QDialog, QFileDialog, QApplication, QMainWindow
from PyQt5.uic import loadUi
from PyQt5 import QtGui
from keras.utils import plot_model


class Application(QMainWindow):

    def __init__(self, data_loader, model_loader, normalizator, title="Dogs detector"):
        super(Application, self).__init__()
        loadUi("gui.ui", self)
        self.data_loader = data_loader
        self.target_size = None
        self.model_loader = model_loader
        self.class_names = None
        self.normalizator = normalizator
        self.initialize_default_values(title)
        self.initialize_buttons()
        self.initialize_radios()
        self.data = None
        self.show()

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select image", ".", filter="Images (*.jpg *.png *.bmp)")

        if path == "":
            return

        self.on_path_updated(path)

    def on_path_updated(self, path):
        self.file_path.setText(path)
        self.data = self.data_loader.load_single_image(path, target_size=self.target_size)
        self.data = self.normalizator.normalize(self.data)
        self.set_label(os.path.basename(os.path.normpath(path)))
        self.set_image(path)

    def set_image(self, path):
        self.image_preview.setPixmap(QtGui.QPixmap(path))

    def set_label(self, label_text):
        self.output_label.setText(label_text)

    def identify_image(self):
        if self.data is None:
            return

        data = self.data
        result = self.model.predict(data)[0]
        self.set_label(self.class_names[result])

    def initialize_default_values(self, title):
        self.setStyleSheet("background-color: rgb(105, 50, 110);"
                           "color: rgb(255, 255, 255);"
                           "font-family: \"Georgia, serif\";")
        self.output_label.setStyleSheet("font-size: 18px;")
        self.setWindowTitle(title)
        self.set_label("")
        self.on_path_updated(os.path.normpath("images/default.jpg"))

    def initialize_buttons(self):
        self.select_image_button.clicked.connect(self.select_image)
        self.select_image_button.setStyleSheet("background-color: rgb(150, 0, 0);")
        self.identify_button.clicked.connect(self.identify_image)
        self.identify_button.setStyleSheet("background-color: rgb(0, 120, 0);")

    def initialize_radios(self):
        self.our_model_button.setChecked(True)
        self.modelGroup.setExclusive(True)
        self.our_model_button.clicked.connect(self.call_our_model)
        self.transfered_model_button.clicked.connect(self.call_transfered_model)
        self.call_our_model()

    def call_our_model(self):
        self.model = self.model_loader.load_model("./newHope/simple")
        self.target_size = (224, 224)
        dirs = sorted(os.listdir("./images/dogs/Images"))
        self.class_names = [dir.split('-', 1)[1].replace("_", " ").capitalize() for dir in dirs]

    def call_transfered_model(self):
        self.model = self.model_loader.load_model("./newHope/transfered")
        self.target_size = (224, 224)
        dirs = sorted(os.listdir("./images/dogs/All_images"))
        self.class_names = [dir.split('-', 1)[1].replace("_", " ").capitalize() for dir in dirs]
