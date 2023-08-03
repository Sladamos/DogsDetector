import sys

from gui.QtApp import QtApp
from option.app.AppOption import AppOption
from PyQt5.QtWidgets import QApplication


class QtAppOption(AppOption):
    def execute(self, config):
        app_config = config["qt"]
        init_config = config["init"]
        data_loader, data_normalizer, detector = self.initialize_app(init_config, app_config)
        app = QApplication(sys.argv)
        my_app = QtApp(data_loader, data_normalizer, detector, app_config)
        my_app.show()
        sys.exit(app.exec_())
