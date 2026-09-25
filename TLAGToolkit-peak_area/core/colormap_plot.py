from cv2 import resize
from PyQt5 import QtGui, QtCore, QtWidgets

import numpy as np
import pyqtgraph as pg
# from pyqtgraph.Qt import QtGui, QtCore

import pandas as pd
from scipy.signal import savgol_filter
from tqdm import tqdm

import fabio

from lmfit import models

import os
import sys

from PyQt5.QtWidgets import QApplication, QGridLayout, QPushButton, QWidget, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap

from PIL import Image
from PyQt5.QtGui import QImage, QPainter


ABSOLUTE_PATH = os.path.dirname(__file__)

class Window(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Colormap")

        self.p_route = False
        self.num_angles = 556
        self.num_images = 100000
        self.sampling_images = 20

        self.time = None
        self.angles = None
        self.max_index = 0

        # ALBA parameters to fasten integrations reading
        self.lines_to_skip = None
        self.header = None

        self.initUI()

    def initUI(self):
        # Create a QGridLayout instance
        layout = QGridLayout()

        self.label_log_values = QLabel()
        self.label_exp_name = QLabel()
        self.label_exp_name.setAlignment(QtCore.Qt.AlignRight)
        self.log_name = ""
        self.integrations_name = ""
        layout.addWidget(self.label_log_values, 0, 0)
        layout.addWidget(self.label_exp_name, 0, 1)

        self.create_trend_plot()
        layout.addWidget(self.p_trend_imv, 1, 0, 1, 2)

        self.create_buttons()
        layout.addLayout(self.hbox_buttons, 2, 0, 1, 2)

        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(1, 1)
        layout.setRowStretch(2, 0)


        self.setLayout(layout)
        self.resize(1000, 750)


    def create_trend_plot(self):

        self.p_integration = pg.PlotWidget()
        self.p_integration.setLabels(bottom='TwoTheta (º)')
        self.p_integration.setLabels(left='Intensity (Arbitrary unit)')

        data1 = 10000 + 15000 * pg.gaussianFilter(np.random.random(size=1000), 10)
        self.p_integration_data = self.p_integration.plot(data1, pen="w")

        # Create baseline curve
        self.p_baseline_integration = self.p_integration.plot()

        # Initialize fitting_array
        self.p_fitting_data = []

        # to display axis ticks inside the ImageView, instantiate it with a PlotItem instance as its view
        self.p_trend_imv = pg.ImageView(view=pg.PlotItem())

        # Hide buttons
        self.p_trend_imv.ui.roiBtn.hide()
        self.p_trend_imv.ui.menuBtn.hide()

        # # Customize axis fonts
        # axis_font = QtGui.QFont()
        # axis_font.setPointSize(13)  # Set the desired font size here

        # self.p_trend_imv.view.getAxis('bottom').setTickFont(axis_font)
        # self.p_trend_imv.view.getAxis('left').setTickFont(axis_font)

        # label_style = {'font-size': '16pt'}  # Set the desired label font size here
        # self.p_trend_imv.view.getAxis('bottom').setLabel('TwoTheta (º)', **label_style)
        # self.p_trend_imv.view.getAxis('left').setLabel('Time (seconds)', **label_style)

        # Add labels
        self.p_trend_imv.view.setLabels(bottom='TwoTheta (º)')
        self.p_trend_imv.view.setLabels(left='Time (seconds)')

        # Modify view
        self.p_trend_imv.view.invertY(False)
        self.p_trend_imv.view.setAspectLocked(False)

        self.p_trend_image = self.p_trend_imv.getImageItem()

        # Easter egg
        image_path = "miscellaneous/group_picture.npy"
        full_path = os.path.join(ABSOLUTE_PATH, image_path)
        data = np.flip(np.load(full_path).transpose())
        data[0][0] = 100

        self.p_trend_imv.setImage(data) 
        
        # Change color map
        cmap = pg.colormap.get("CET-R4")
        self.p_trend_imv.setColorMap(cmap)

        # Hide ticks from ImageView histogram
        self.p_trend_imv.ui.histogram.gradient.showTicks(False)

        # Cross hairs
        self.hLine_trend = pg.InfiniteLine(angle=0, movable=False)
        self.p_trend_imv.addItem(self.hLine_trend, ignoreBounds=True)

        self.vLine_trend = pg.InfiniteLine(angle=90, movable=False)
        self.p_trend_imv.addItem(self.vLine_trend, ignoreBounds=True)



    def create_buttons(self):
        self.hbox_buttons = QtWidgets.QHBoxLayout()

        button_integration = QtWidgets.QPushButton("Load integrated images")
        button_save_image = QtWidgets.QPushButton("Save image")

        button_integration.clicked.connect(lambda: self.openFileNameDialog("integration"))
        button_save_image.clicked.connect(lambda: self.openFileNameDialog("save image"))

        self.hbox_buttons.addWidget(button_integration)
        self.hbox_buttons.addWidget(button_save_image)


    def openFileNameDialog(self, button):
        if button == "save image":
            name = "Save image"
            available_extensions = "PNG files (*.png);;JPEG files (*.jpg *.jpeg);;All files (*.*)"
            filename, _ = QFileDialog.getSaveFileName(self, name, "", available_extensions)

            if filename:
                
                scaling_factor = 5     # Specify the scaling factor for higher resolution
                
                original_size = self.p_trend_imv.size()
                
                # Create a larger QImage based on the scaling factor
                image = QImage(original_size.width() * scaling_factor, 
                            original_size.height() * scaling_factor, 
                            QImage.Format_ARGB32)
                image.setDevicePixelRatio(scaling_factor)
                
                painter = QPainter(image)
                self.p_trend_imv.render(painter)
                painter.end()
                

                image.save(filename)



            # current_image = self.grab()
            # current_image.save(filename)

        if button == "integration":
            name = "Load integrated files"
            available_extensions = "text files (*.txt *.dat)"
            filenames, _ = QFileDialog.getOpenFileNames(self,name, "",available_extensions)
            self.update_integrations(filenames)


    def read_integrations_alba(self, filename):
        if self.lines_to_skip is None:
            with open(filename) as f:
                line = f.readline()
                cnt = 0
                while line.startswith('#'):
                    prev_line = line
                    line = f.readline()
                    cnt += 1

            self.lines_to_skip = cnt
            self.header = prev_line.strip().lstrip('# ').split()

        df = pd.read_csv(filename, delimiter="\s+",
                        names=self.header,
                        skiprows=self.lines_to_skip)

        return df


    def update_label_exp_name(self):
        self.label_exp_name.setText(", ".join([self.log_name, self.integrations_name]))

    def update_integrations(self, filenames):
        self.num_images = len(filenames)

        self.alba = False
        if filenames[0].split(".")[-1] == "dat":
            self.alba = True

        if self.alba:
            self.num_angles = 2880 # 1000 for March 2880 for June
            # Get integration_name
            integration_filename = os.path.basename(os.path.normpath(filenames[0]))
            self.integrations_name = "_".join(integration_filename.split("_")[1:3])
        else:
            self.num_angles = 557
            # Get integration_name
            self.integrations_name = os.path.basename(os.path.split(filenames[0])[0])

        self.update_label_exp_name()

        self.integrations = {}

        for i, filename in tqdm(enumerate(filenames), total=self.num_images, desc="Loading integration files"):
            index = int(filename.split(".")[0].split("_")[-1])
            self.max_index = max(self.max_index, index)
            if self.alba:
                df = self.read_integrations_alba(filename)
                self.integrations[index] = df["I"]
            else:
                df = pd.read_csv(filename, sep = ' ')
                self.integrations[index] = df["intensity"]

        if self.alba:
            self.angles = df["2th_deg"].values
        else:
            self.angles = df["#twoTh"].values

        # Plot first integration
        self.p_integration_data.setData(self.angles, next(iter(self.integrations.values())), pen="w")
        self.p_integration.autoRange()

        self.plot_image_integrations_trend()


    def plot_image_integrations_trend(self):
        new_integrations = np.array([v for v in self.integrations.values()])

        # Add image of the integrations trend
        self.p_trend_imv.setImage(new_integrations.transpose())

        # Adjust axes intervals
        height = self.max_index
        initial_time = 0

        width = max(self.angles) - min(self.angles)
        self.p_trend_image.setRect(QtCore.QRectF(min(self.angles), initial_time, width, height))
        self.p_trend_imv.autoRange()


def main():
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
