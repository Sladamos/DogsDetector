import os

from PyQt5.QtWidgets import QDialog, QFileDialog, QApplication, QMainWindow
from PyQt5.uic import loadUi
from PyQt5 import QtGui


class Application(QMainWindow):

    def __init__(self, data_loader, model, normalizator, class_names, title="Dogs detector"):
        super(Application, self).__init__()
        # TODO: pretty gui
        loadUi("gui.ui", self)
        self.data_loader = data_loader
        self.data = None
        self.model = model
        self.normalizator = normalizator
        self.class_names = class_names

        self.initialize_default_values(title)
        self.initialize_buttons()
        self.show()

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select image", filter="Images (*.jpg *.png *.bmp)")

        if path == "":
            return

        self.on_path_updated(path)

    def on_path_updated(self, path):
        self.file_path.setText(path)
        self.data = self.data_loader.load_single_image(path)
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
        self.data = None

    def initialize_default_values(self, title):
        self.setStyleSheet("background-color: rgb(20, 20, 20);"
                           "color: rgb(255, 255, 255);"
                           "font-family: \"Comic Sans MS\", \"Comic Sans\", cursive;")
        self.output_label.setStyleSheet("font-size: 18px;")
        self.setWindowTitle(title)
        self.set_label("")
        self.on_path_updated(os.path.normpath("images/default.jpg"))

    def initialize_buttons(self):
        self.select_image_button.clicked.connect(self.select_image)
        self.select_image_button.setStyleSheet("background-color: rgb(140, 0, 0);")
        self.identify_button.clicked.connect(self.identify_image)
        self.identify_button.setStyleSheet("background-color: rgb(0, 140, 0);")
