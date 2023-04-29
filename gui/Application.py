from PyQt5.QtWidgets import QDialog, QFileDialog, QApplication, QMainWindow
from PyQt5.uic import loadUi
from PyQt5.uic.properties import QtGui


class Application(QMainWindow):

    def __init__(self):
        super(Application, self).__init__()
        loadUi("gui.ui", self)

        # Set default stuff here
        # --------------------------

        self.setWindowTitle("Dogs detector")
        self.out_label.setText("")
        #self.image_preview.setPixmap(QtGui.QPixmap("images/default.png"))

        # --------------------------

        self.setStyleSheet("background-color: rgb(20, 20, 20);"
                           "color: rgb(255, 255, 255);"
                           "font-family: \"Comic Sans MS\", \"Comic Sans\", cursive;")
        self.out_label.setStyleSheet("font-size: 18px;")
        self.img_path = ""

        #self.select_button.clicked.connect(self.select_file)
        self.select_image_button.setStyleSheet("background-color: rgb(140, 0, 0);")
        #self.identify_button.clicked.connect(self.process)
        self.detect_dog_button.setStyleSheet("background-color: rgb(0, 140, 0);")


        self.show()
