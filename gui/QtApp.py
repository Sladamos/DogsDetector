import os
import sys

from PyQt5.QtWidgets import QDialog, QFileDialog, QApplication, QMainWindow
from PyQt5.uic import loadUi
from PyQt5 import QtGui
from keras.utils import plot_model

from gui.App import App


class QtApp(QMainWindow, App):

    def __init__(self, data_loader, data_normalizer, detector):
        super(QMainWindow, self).__init__()
        App.__init__(self, data_loader, data_normalizer, detector)
        loadUi("gui.ui", self)
        self.initialize_default_values("Dogs detector")
        self.initialize_buttons()
        self.initialize_radios()

    def select_image_path(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select image", ".", filter="Images (*.jpg *.png *.bmp)")

        if path == "":
            return

        self.on_path_updated(path)

    def on_path_updated(self, path):
        self.file_path.setText(path)
        self.load_image(path)
        self.set_label(os.path.basename(os.path.normpath(path)))
        self.set_image(path)

    def set_image(self, path):
        self.image_preview.setPixmap(QtGui.QPixmap(path))

    def set_label(self, label_text):
        self.output_label.setText(label_text)

    def identify_image(self):
        if self.is_image_selected():
            result = self.get_classification_result()
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
        self.select_image_button.clicked.connect(self.select_image_path)
        self.select_image_button.setStyleSheet("background-color: rgb(150, 0, 0);")
        self.identify_button.clicked.connect(self.identify_image)
        self.identify_button.setStyleSheet("background-color: rgb(0, 120, 0);")

    def initialize_radios(self):
        self.our_model_button.setChecked(True)
        self.modelGroup.setExclusive(True)
        self.our_model_button.clicked.connect(self.select_simple_model)
        self.transfered_model_button.clicked.connect(self.select_transfered_model)

