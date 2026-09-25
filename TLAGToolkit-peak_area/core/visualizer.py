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

from .utils import BASELINES_FUNCTIONS

import os
import sys

from PyQt5.QtWidgets import (
    QApplication,
    QGridLayout,
    QPushButton,
    QWidget,
    QLabel,
    QFileDialog
)

ABSOLUTE_PATH = os.path.dirname(__file__)

class Window(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TLAG Visualizer")

        self.p_route = False
        self.num_angles = 556
        self.num_images = 100000
        self.sampling_images = 20

        self.temperature = None
        self.pressure = None
        self.time = None
        self.angles = None
        self.logs_loaded = False
        self.max_index = 0

        # ALBA parameters to fasten integrations reading
        self.lines_to_skip = None
        self.header = None

        self.initUI()

    def initUI(self):
        # Create a QGridLayout instance
        layout = QGridLayout()
        # pg.dbg()
        # Add widgets to the layout

        self.label_log_values = QLabel()
        self.label_exp_name = QLabel()
        self.label_exp_name.setAlignment(QtCore.Qt.AlignRight)
        self.log_name = ""
        self.integrations_name = ""
        # self.label.addItem(pg.LabelItem(justify='right'))
        layout.addWidget(self.label_log_values, 0, 0)
        layout.addWidget(self.label_exp_name, 0, 1)

        self.create_temperature_plot()
        layout.addWidget(self.p_temp, 1, 1)

        if self.p_route:
            self.add_pressure_plot()

        self.create_integrations_plot()
        layout.addWidget(self.p_integration, 1, 0)

        self.create_images_plot()
        layout.addWidget(self.p_image, 2, 0)

        self.create_trend_plot()
        layout.addWidget(self.p_trend_imv, 2, 1)

        self.create_buttons()
        layout.addLayout(self.hbox_buttons, 3, 0, 1, 2)

        self.create_configurations()
        layout.addLayout(self.hbox_configurations, 4, 0, 1, 2)

        # Set the layout on the application's window
        self.setLayout(layout)


    def interaction_crosshairs(self, mousePoint_time, mousePoint_theta=None):
        
        if self.time is not None:
            if (self.time < mousePoint_time).any():
                index = np.where(self.time == self.time[self.time < mousePoint_time][-1])[0][0]
            else:
                # Do nothing
                return
        else:
            index = int(mousePoint_time)

        if index > 0 and index < self.max_index:
            # pass
            if self.pressure is not None:
                current_pressure = round(self.pressure[index], 4)
            else:
                current_pressure = "-"
            
            if self.temperature is not None:
                current_temperature = round(self.temperature[index], 2)
            else:
                current_temperature = "-"

            if self.time is not None:
                current_time = round(mousePoint_time, 1)
            else:
                current_time = "-"
            
            text_temperature = f"<span style='font-size: 12pt'><span style='color: red'>Temperature={current_temperature} ºC, </span> </span>"
            text_pressure = f"<span style='font-size: 12pt'><span style='color: DeepSkyBlue'>Pressure={current_pressure} mbar</span> </span>"
            text_time = f"<span style='font-size: 12pt'><span style='color: green'>Time={current_time} seconds</span> </span>"
            text_index = f"<span style='font-size: 12pt'><span style='color: gray'>Index={index} </span> </span>"
            self.label_log_values.setText(text_temperature + text_pressure + text_time + text_index) #(data1[index], data2[index])
        
            # Update integrations
            try:
                closest_index_index = Window.findClosest(list(self.integrations.keys()), index)
                closest_index = list(self.integrations.keys())[closest_index_index]
                self.p_integration_data.setData(self.angles, self.integrations[closest_index], pen="w")
                
                # Test filtered integration
                # smoothed_2dg = savgol_filter(self.integrations[index], window_length = 7, polyorder = 2, mode = "constant")
                # self.p_baseline_integration.setData(self.angles, smoothed_2dg, pen="g")

            except:
                pass
            
            # Update diffraction image
            try:
                self.img_diffraction.setImage(self.diff_images[closest_index//self.sampling_images], autoLevels=False)
            except:
                pass                                

            # Update baseline
            if self.dropdown_baseline.currentText() != "None":
                try:
                    # baseObj=BaselineRemoval(self.integrations[index])
                    # y_corrected=baseObj.IModPoly(degree=2)
                    # baseline = self.integrations[index] - y_corrected

                    baseline = BASELINES_FUNCTIONS[self.dropdown_baseline.currentText()](data=self.integrations[closest_index])

                    self.p_baseline_integration.setData(self.angles, baseline, pen="c")
                except Exception as e:
                    print("ERROR baseline", e)
            else:
                self.p_baseline_integration.clear()
            
            # Not working for python 3.5
            # Update fittings
            try:
                index_fit = closest_index
                if index_fit >= self.min_index_fit and index_fit <= self.max_index_fit:
                    filtered_df_fit = self.df_fit[self.df_fit["imgIndex"] <= index_fit]

                    try:
                        filtered_df_fit["imgIndex"] == index_fit
                    except:
                        print("Fitting not found for current index")
                        raise Exception
                    
                    for i, (prefix, color) in enumerate(zip(self.prefixes, self.color_fits)):
                        # self.p_fitting_data[i].setData(filtered_df_fit[f"{prefix}_center"], filtered_df_fit[f"{prefix}_height"] + 2, pen=color)

                        columns_current_model = list(filter(lambda x: x.split('_')[0] == prefix, self.df_fit.columns))

                        # Determine model with the current params. Currently it can differentiat GaussianModel, VoigtModel, PseudoVoigModel
                        model_used = Window.get_model(columns_current_model)

                        # Plot peak fit
                        model = getattr(models, model_used)(prefix=prefix + '_')
                        
                        params = {column : filtered_df_fit[filtered_df_fit["imgIndex"] == index_fit][column].values[0] for column in columns_current_model}

                        params_model = model.make_params(**params)

                        fitted_peak = model.eval(params_model, x=self.angles)

                        try:
                            fitted_peak_baseline = fitted_peak + baseline
                        except:
                            fitted_peak_baseline = fitted_peak

                        peak_index = Window.findClosest(self.angles, params[f"{prefix}_center"])
                        interval = 30
                        cropped_fitted_peak_baseline = fitted_peak_baseline[max(0, peak_index - interval) : peak_index + interval]
                        cropped_angles = self.angles[max(0, peak_index - interval) : peak_index + interval]

                        self.p_fitting_data[i].setData(cropped_angles, cropped_fitted_peak_baseline, pen=color)

            except Exception as e:
                pass

            # Finally update vertical lines
            self.vLine_logs.setPos(mousePoint_time)
            self.hLine_trend.setPos(mousePoint_time)            
                
        elif self.max_index == 0:
            # if self.max_index hasn't been updated, update the vertical lines as well
            self.vLine_logs.setPos(mousePoint_time)
            self.hLine_trend.setPos(mousePoint_time)

        if mousePoint_theta is not None:
            self.vLine_trend.setPos(mousePoint_theta)


    def mouseMoved_temperature(self, evt):
        pos = evt  ## using signal proxy turns original arguments into a tuple
        if self.p_temp.sceneBoundingRect().contains(pos):
            mousePoint = self.p_temp.plotItem.vb.mapSceneToView(pos)
            self.interaction_crosshairs(mousePoint.x())


    def mouseMoved_trend(self, evt):
        pos = evt  ## using signal proxy turns original arguments into a tuple
        if self.p_trend_image.sceneBoundingRect().contains(pos):
            mousePoint = self.p_trend_imv.view.vb.mapSceneToView(pos)
            self.interaction_crosshairs(mousePoint.y(), mousePoint.x())


    def create_temperature_plot(self):

        self.p_temp = pg.PlotWidget()
        self.p_temp.setLabels(left='Temperature (ºC)')
        self.p_temp.setAutoVisible(y=True) # Do not know what this does

        self.p_temp.setLabels(bottom='Time (seconds)')

        # self.p_temp.vb.sigResized.connect(updateViews)

        data1 = 5000 * pg.gaussianFilter(np.random.random(size=1000), 10)
        self.temperature_ploted = self.p_temp.plot(data1, pen="r")

        # Cross hair
        self.vLine_logs = pg.InfiniteLine(angle=90, movable=False)
        self.p_temp.addItem(self.vLine_logs, ignoreBounds=True)

        # Add cross hair interaction
        self.p_temp.scene().sigMouseMoved.connect(slot=self.mouseMoved_temperature)

    def updateViews(self):
        # Update views if we are also plotting the pressure
        ## view has resized; update auxiliary views to match
        self.p_pressure.setGeometry(self.p_temp.plotItem.vb.sceneBoundingRect())
        
        ## need to re-update linked axes since this was called
        ## incorrectly while views had different shapes.
        ## (probably this should be handled in ViewBox.resizeEvent)
        self.p_pressure.linkedViewChanged(self.p_temp.plotItem.vb, self.p_pressure.XAxis)

    def add_pressure_plot(self):
        self.p_pressure = pg.ViewBox()
        self.p_temp.showAxis('right')
        self.p_temp.setLabels(right='Total Pressure (mbar)')
        self.p_temp.scene().addItem(self.p_pressure)
        self.p_temp.getAxis('right').linkToView(self.p_pressure)
        self.p_pressure.setXLink(self.p_temp)

        self.updateViews()
        self.p_temp.plotItem.vb.sigResized.connect(self.updateViews)

    def create_integrations_plot(self):
        self.p_integration = pg.PlotWidget()
        self.p_integration.setLabels(bottom='TwoTheta (º)')
        self.p_integration.setLabels(left='Intensity (Arbitrary unit)')

        data1 = 10000 + 15000 * pg.gaussianFilter(np.random.random(size=1000), 10)
        self.p_integration_data = self.p_integration.plot(data1, pen="w")

        # Create baseline curve
        self.p_baseline_integration = self.p_integration.plot()

        # Initialize fitting_array
        self.p_fitting_data = []
        

    def create_images_plot(self):
        self.p_image = pg.PlotWidget()
        self.p_image.setLabels(bottom='Orientative TwoTheta (º)')

        self.img_diffraction = pg.ImageItem()
        self.p_image.addItem(self.img_diffraction)

        data = np.random.normal(size=(200, 100))
        data[20:80, 20:80] += 2.
        data = pg.gaussianFilter(data, (3, 3))
        data += np.random.normal(size=(200, 100)) * 0.1
        self.img_diffraction.setImage(data)

        cm = pg.colormap.get("CET-R4")
        lut = cm.getLookupTable(0.0, 1.0)
        levels = [1,3]
        self.img_diffraction.setLookupTable(lut)
        self.img_diffraction.setLevels(levels)
        self.p_image.getPlotItem().hideAxis('left')


    def create_trend_plot(self):

        # to display axis ticks inside the ImageView, instantiate it with a PlotItem instance as its view
        self.p_trend_imv = pg.ImageView(view=pg.PlotItem())

        # Hide buttons
        self.p_trend_imv.ui.roiBtn.hide()
        self.p_trend_imv.ui.menuBtn.hide()

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

        # Add cross hair interaction
        self.p_trend_imv.scene.sigMouseMoved.connect(slot=self.mouseMoved_trend)


    def create_buttons(self):
        self.hbox_buttons = QtWidgets.QHBoxLayout()
        button_logs = QtWidgets.QPushButton("Load log file")
        button_integration = QtWidgets.QPushButton("Load integrated images")
        button_images = QtWidgets.QPushButton("Load diffraction images")
        button_fitting = QtWidgets.QPushButton("Load fitting")

        button_logs.clicked.connect(lambda: self.openFileNameDialog("logs"))
        button_integration.clicked.connect(lambda: self.openFileNameDialog("integration"))
        button_images.clicked.connect(lambda: self.openFileNameDialog("images"))
        button_fitting.clicked.connect(lambda: self.openFileNameDialog("fitting"))


        self.hbox_buttons.addWidget(button_logs)
        self.hbox_buttons.addWidget(button_integration)
        self.hbox_buttons.addWidget(button_images)
        self.hbox_buttons.addWidget(button_fitting)

    def create_configurations(self):
        self.hbox_configurations = QtWidgets.QHBoxLayout()
        self.dropdown_baseline = QtWidgets.QComboBox(self)
        self.checkbox_baseline = QtWidgets.QCheckBox("Apply baseline to trend plot")

        self.dropdown_baseline.addItems(BASELINES_FUNCTIONS.keys())
        self.dropdown_baseline.activated.connect(self.plot_image_integrations_trend)
        self.checkbox_baseline.stateChanged.connect(self.plot_image_integrations_trend)

        self.hbox_configurations.addWidget(self.dropdown_baseline)
        self.hbox_configurations.addWidget(self.checkbox_baseline)

    def openFileNameDialog(self, button):
        if button == "logs":
            name = "Load log file"
            available_extensions = "text files (*.txt *.log)"
            filename, _ = QFileDialog.getOpenFileName(self,name, "",available_extensions)
            self.update_logs(filename)
        if button == "integration":
            name = "Load integrated files"
            available_extensions = "text files (*.txt *.dat)"
            filenames, _ = QFileDialog.getOpenFileNames(self,name, "",available_extensions)
            self.update_integrations(filenames)
        if button == "images":
            name = "Load diffraction image files"
            available_extensions = "raw files (*.raw *.edf)"
            filenames, _ = QFileDialog.getOpenFileNames(self,name, "",available_extensions)
            self.update_images(filenames)
        if button == "fitting":
            name = "Load peak fitting"
            available_extensions = "csv file (*.csv)"
            filenames, _ = QFileDialog.getOpenFileName(self,name, "",available_extensions)
            self.update_fitting(filenames)


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
                        skiprows=self.lines_to_skip
                    )

        return df

    def update_logs(self, filename):
        # Remove previous plots
        self.p_temp.removeItem(self.temperature_ploted)
        if self.p_route:
            self.p_pressure.clear()

        self.log_name = os.path.basename(os.path.normpath(filename))
        self.update_label_exp_name()

        df = pd.read_csv(filename, sep=',')
        self.max_index = df["imgIndex"].max()
        self.temperature = dict(zip(df["imgIndex"], df["temperature"]))
        self.time = np.array(df["time"])

        self.temperature_ploted = self.p_temp.plot(self.time, df["temperature"], pen="r")

        if not df["pressure"].isna().any():
            self.p_route = True
            # self.pressure = np.array(df["pressure"])
            self.pressure = dict(zip(df["imgIndex"], df["pressure"]))
            self.add_pressure_plot()
            self.p_pressure.addItem(pg.PlotCurveItem(self.time, list(df["pressure"]), pen='b'))
        else:
            self.p_route = False
            try:
                self.p_pressure.clear()
            except:
                pass
            self.p_temp.hideAxis('right')

        self.logs_loaded = True

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

        # self.integrations = np.zeros((self.num_images, self.num_angles))
        self.integrations = {}

        for i, filename in tqdm(enumerate(filenames), total=self.num_images, desc="Loading integration files"):
            index = int(filename.split(".")[0].split("_")[-1])
            if not self.logs_loaded:
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
        if self.checkbox_baseline.isChecked() and self.dropdown_baseline.currentText() != "None":
            first_integration = next(iter(self.integrations.values()))
            reference_baseline = BASELINES_FUNCTIONS[self.dropdown_baseline.currentText()](first_integration)
            new_integrations = np.array([integration-reference_baseline for integration in self.integrations.values()])
        else:
            new_integrations = np.array([v for v in self.integrations.values()])

        # Add image of the integrations trend
        self.p_trend_imv.setImage(new_integrations.transpose())

        # Adjust axes intervals
        # If logs are not loaded, height is indices instead of time
        if self.logs_loaded:
            height = (max(self.time) + (self.time[-1] - self.time[-2])) - min(self.time)
            initial_time = min(self.time)
        else:
            height = self.max_index
            initial_time = 0

        width = max(self.angles) - min(self.angles)
        self.p_trend_image.setRect(QtCore.QRectF(min(self.angles), initial_time, width, height))
        self.p_trend_imv.autoRange()


    def update_images(self, filenames):
        self.num_images = len(filenames)
        self.num_images_sampled = self.num_images//self.sampling_images + 1
        if self.alba:
            low = 1100
            upper = 2200
            size_theta = 560
            size_phi = 240
            self.diff_images = np.zeros((self.num_images_sampled, size_theta, size_phi), dtype="float32")

            for i, filename in tqdm(enumerate(filenames), total=self.num_images, desc="Loading image files"):
                if i % self.sampling_images != 0:
                    continue
                A = fabio.open(filename)
                A_cropped = A.data[low:upper,:]
                A_resized = resize(A_cropped, (size_phi, size_theta))[...,::-1,:]
                # .swapaxes(-2,-1)[...,::-1,:]
                self.diff_images[i//self.sampling_images] = np.log10(A_resized + 1)
        else:
            self.diff_images = np.zeros((self.num_images_sampled, 560, 240))
            for i, filename in tqdm(enumerate(filenames), total=self.num_images_sampled, desc="Loading image files"):
                if i % self.sampling_images != 0:
                    continue
                A = np.fromfile(filename, dtype = 'uint32', sep="")
                A = A.reshape([240, 560])
                self.diff_images[i//self.sampling_images] = np.log10(A + 1).swapaxes(-2,-1)[...,::-1,:]
        
        # Plot first image
        self.img_diffraction.setImage(self.diff_images[0], autoLevels=True)
        width = max(self.angles) - min(self.angles)
        self.img_diffraction.setRect(QtCore.QRectF(min(self.angles), 0, width, 1))

    def update_fitting(self, filename):
        self.df_fit = pd.read_csv(filename)

        columns = self.df_fit.columns
        
        logs_desired = ["imgIndex", "temperature", "time", "pressure", "resistivity", "timestamp"]
        self.prefixes = set([column.split("_")[0] for column in columns if column not in logs_desired])
        print("Prefixes found: ", self.prefixes)
        
        if len(self.p_fitting_data) > 0:
            for i in range(len(self.p_fitting_data)):
                self.p_integration.removeItem(self.p_fitting_data[i])
            
        self.p_fitting_data = [self.p_integration.plot() for i in range(len(self.prefixes))]

        self.color_fits = ["r", "g", "y", "b", "o"]

        self.min_index_fit = self.df_fit["imgIndex"].min()
        self.max_index_fit = self.df_fit["imgIndex"].max()

    # Returns element closest to target in arr[]
    @staticmethod
    def findClosest(arr, target):

        n = len(arr)
        # Corner cases
        if (target <= arr[0]):
            return 0
        if (target >= arr[n - 1]):
            return n-1

        # Doing binary search
        i = 0; j = n; mid = 0
        while (i < j): 
            mid = (i + j) // 2

            if (arr[mid] == target):
                return arr[mid]

            # If target is less than array 
            # element, then search in left
            if (target < arr[mid]) :

                # If target is greater than previous
                # to mid, return closest of two
                if (mid > 0 and target > arr[mid - 1]):
                    # return getClosest(arr[mid - 1], arr[mid], target)
                    return mid - 1

                # Repeat for left half 
                j = mid
            
            # If target is greater than mid
            else :
                if (mid < n - 1 and target < arr[mid + 1]):
                    # return getClosest(arr[mid], arr[mid + 1], target)
                    return mid + 1
                    
                # update i
                i = mid + 1
            
        # Only single element left after search
        # return arr[mid]
        return mid

    @staticmethod
    def get_model(model_params):
        params = [column.split("_")[-1] for column in model_params]

        if 'gamma' in params:
            if 'fraction' in params:
                return "PseudoVoigtModel"
            else:
                return "VoigtModel"
        else:
            return "GaussianModel" 


def main():
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
