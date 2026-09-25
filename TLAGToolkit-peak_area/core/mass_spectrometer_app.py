# -*- coding: utf-8 -*-
"""
ANALYSING MASS SPECTROMETER - CO2 EMITTED SIGNAL 

Last version: 15/08/2025

@author: omola
"""

import tkinter as tk
from tkinter import filedialog
import os
import csv
from scipy.interpolate import pchip
import statistics
import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, uic
import pyqtgraph
from scipy.interpolate import UnivariateSpline




class CO2_fitting_app(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        ui_file_path = "C:\PhD_C\DATA_ANALYSIS\PYTHON_PROGRAMS\TLAGToolkit_ALBA\mass_spectrometer_CO2.ui"
        uic.loadUi(ui_file_path, self)  # Load the .ui file

        # Prepare plot
        self.create_plot_widget()
        
        # Generate some synthetic data
        self.x_data = []
        self.y_data = []
        
        # Add curves to the plot
        self.data_curve = self.app_plot.plot(pen='b', name="Data")
        self.fit_curve = self.app_plot.plot(pen='r', name="Background")
        self.start_marker = self.app_plot.plot(pen=None, symbol='o', symbolBrush='m', symbolSize=10)
        self.end_marker = self.app_plot.plot(pen=None, symbol='o', symbolBrush='m', symbolSize=10)
        self.jump_marker = self.app_plot.plot(pen=None, symbol='o', symbolBrush='g', symbolSize=10)
        self.data_curve.setData(self.x_data, self.y_data)

        # Connect upload button
        self.upload_data.clicked.connect(self.upload_multiple_csv_files)
        self.download_data.clicked.connect(self.download_fitting)

        # Connect parameter changes to update plot
        self.start_time.valueChanged.connect(self.update_fit)
        self.end_time.valueChanged.connect(self.update_fit)
        self.smooth_num.valueChanged.connect(self.update_fit)
        self.smooth_power.valueChanged.connect(self.update_fit)

        # Initial plot
        self.update_fit()

    def create_plot_widget(self):
        self.app_plot = pyqtgraph.PlotWidget()
        self.plot = self.plot_layout.addWidget(self.app_plot)

    def update_fit(self):
        if len(self.x_data) < 2:
            return  # No data loaded yet
        
        self.x_data_fit = self.x_data[1:self.idx_jump]
        self.y_data_fit = self.y_data[1:self.idx_jump]

        start_time = self.start_time.value()
        end_time = self.end_time.value()
        smooth_num = self.smooth_num.value()
        smooth_power = int(self.smooth_power.value())

        # Find background regions
        idx_start = self.find_index_of_closest_num(self.x_data_fit, start_time)
        idx_end = self.find_index_of_closest_num(self.x_data_fit, end_time)
        background_region = (self.x_data_fit < start_time) | (self.x_data_fit > end_time)

        x_background = self.x_data_fit[background_region]
        y_background = self.y_data_fit[background_region]

        # Update start/end markers
        self.start_marker.setData([self.x_data_fit[idx_start]], [self.y_data_fit[idx_start]])
        self.end_marker.setData([self.x_data_fit[idx_end]], [self.y_data_fit[idx_end]])
        
        # Compute spline fit
        try:
            smooth_value = smooth_num/1000*(10**smooth_power)
            spline = UnivariateSpline(x_background, y_background, k=3, s=smooth_value)
            self.background = spline(self.x_data_fit)
            self.fit_curve.setData(self.x_data_fit, self.background)
        except Exception as e:
            print("Spline fitting failed:", e)
            self.fit_curve.clear()



    def upload_multiple_csv_files(self):
        root = tk.Tk()
        root.withdraw()
        csv_files = filedialog.askopenfilenames(
            title='Select CSV files',
            filetypes=(('CSV files', '*.csv'),),
            multiple=False
        )

        if not csv_files:
            QtWidgets.QMessageBox.warning(self, "No files", "Please select at least one CSV file.")
            return

        self.file = csv_files[0]
        self.directory = os.path.dirname(self.file)
        print("Selected files:", csv_files)


        # TODO: load your data into self.x_data, self.y_data here
        # Example: load first CSV just for testing
        try:
            self.data = self.read_mass_spectrometer_data()
            self.x_data, self.y_data = np.array(self.data['time']), np.array(self.data['CO2'])
            
            # After loading data (self.x_data)
            x_min = min(self.x_data)
            x_max = max(self.x_data)

            self.start_time.setMinimum(x_min)
            self.start_time.setMaximum(x_max)
            self.end_time.setMinimum(x_min)
            self.end_time.setMaximum(x_max)

            # Set step to match your data spacing (if uniform)
            step = np.min(np.diff(self.x_data))  # smallest difference between points
            self.start_time.setSingleStep(step)
            self.end_time.setSingleStep(step)

            self.data_curve.setData(self.x_data, self.y_data)
            self.find_jump()
            self.update_fit()
            if self.idx_jump is not None:
                self.jump_marker.setData([self.x_data[self.idx_jump]], [self.y_data[self.idx_jump]])
            else:
                self.jump_marker.clear()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load CSV: {e}")


    def read_mass_spectrometer_data(self):
        with open(self.file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for _ in range(40):    # Skip the first 38 lines
                next(csvfile)

            data = {
                'time': [],
                'H2': [],
                'H2O': [],
                'CO-N2': [],
                'O2': [],
                'Ar': [],
                'CO2': []
            }
            for row in reader:
                data['time'].append(self.time_to_seconds(row[1]))
                data['H2'].append(float(row[6]))
                data['H2O'].append(float(row[7]))
                data['CO-N2'].append(float(row[8]))
                data['O2'].append(float(row[9]))
                data['Ar'].append(float(row[10]))
                data['CO2'].append(float(row[11]))
        return data

    def time_to_seconds(self, time_str):
        hours, minutes, seconds = time_str.split(':')
        total_seconds = int(hours)*3600 + int(minutes)*60 + float(seconds)
        return round(total_seconds,3)

    def find_jump(self):
        # find moment where we close the masss spectrometer, so the jump    
        interpolation = pchip(self.x_data,self.y_data)
        x_interpolation = np.arange(self.x_data[0],self.x_data[-1], 0.1)
        y_interpolation = interpolation(x_interpolation)
        ms_derivative = self.calculate_derivative(x_interpolation, y_interpolation)
        jump_ms_time = x_interpolation[np.argmin(ms_derivative)]
        index_jump_ms = self.find_index_of_closest_num(self.x_data, jump_ms_time) - 1
        
        criteria_for_deciding_that_jump_exists = statistics.stdev(ms_derivative)*8
        abs_value_derivative_jump = abs(min(ms_derivative))
        
        if abs_value_derivative_jump < criteria_for_deciding_that_jump_exists:
            index_jump_ms = None
            
        self.idx_jump = index_jump_ms

    def calculate_derivative(self, x_data, y_data):
        dx = np.diff(x_data)
        dy = np.diff(y_data)
        derivative = dy / dx
        return derivative


    def find_index_of_closest_num(self, list, target_num):
        ''' given a target number, it finds the index of the closest number on a certain list. 
            this list is meant to be ordered, since the algorithm is optimized for this case'''
        low = 0
        high = len(list) - 1
        closest_index = None

        while low <= high:
            mid = (low + high) // 2
            if list[mid] == target_num:
                return mid
            elif list[mid] < target_num:
                low = mid + 1
            else:
                high = mid - 1
            if closest_index is None or abs(list[mid] - target_num) < abs(list[closest_index] - target_num):
                closest_index = mid
        return closest_index



    def download_fitting(self):
        # PLOT 1: data + background fitting
        fig, ax = plt.subplots(tight_layout={'pad': 0.5})
        ax.plot(self.x_data, self.y_data, '-', color='blue', markersize=2, label='mass spectrometer')
        ax.plot(self.x_data_fit, self.background, '--', color='red', label='background')
        ax.legend(loc='upper left')
        ax.set_ylabel('counts')
        ax.set_xlabel('time (s)')
        figure_name = self.file[:-4] + '_ms1'
        plt.savefig(figure_name + '.png', dpi=500, bbox_inches='tight')
        
        # PLOT 2: data with background subtracted
        self.data['time_clean'] = self.x_data_fit
        self.data['CO2_clean'] = self.y_data_fit - self.background
        
        fig, ax = plt.subplots(tight_layout={'pad': 0.5})
        ax.plot(self.data['time_clean'], self.data['CO2_clean'], '-', color='blue', markersize=2)
        ax.legend(loc='upper left')
        ax.set_ylabel('counts')
        ax.set_xlabel('time (s)')
        figure_name = self.file[:-4] + '_ms2'
        plt.savefig(figure_name + '.png', dpi=500, bbox_inches='tight')
        
        # PLOT 3: INTEGRATED COUNTS
        self.integrate_CO2_data()
        
        fig, ax = plt.subplots(tight_layout={'pad': 0.5})
        ax.plot(self.data['time_int'], list(self.data['CO2_int']))
        ax.legend(loc='upper left')
        ax.set_ylabel('integrated counts')
        ax.set_xlabel('time (s)')
        figure_name = self.file[:-4] + '_ms3'
        plt.savefig(figure_name + '.png', dpi=500, bbox_inches='tight')
        
        self.save_dict_to_csv()
        
        
    def integrate_CO2_data(self):
        x_data = self.data['time_clean']
        y_data = self.data['CO2_clean']
        
        y_data_int = []
        accumulative_int = 0
        for i in range(len(x_data) - 2):
            area_i = (x_data[i+1]-x_data[i]) * (y_data[i+1]+y_data[i])/2
            accumulative_int = accumulative_int + area_i
            y_data_int.append(accumulative_int)
        self.data['CO2_int'] = y_data_int
            
        x_data_int = []
        for i in range(len(x_data) - 2):  # Loop until the second last element
            x_data_int.append((x_data[i+1]-x_data[i])/2 + x_data[i])
        self.data['time_int'] = x_data_int
        
    def pad_lists_to_same_length(self, data):
        max_length = max(len(lst) for lst in data.values())
        for key, lst in data.items():
            if isinstance(lst, np.ndarray):
                lst = lst.tolist()
            if len(lst) < max_length:
                data[key] = lst + [np.nan] * (max_length - len(lst))
        return data

    def save_dict_to_csv(self):
        path_save_file = self.file[:-4] + '_CO2_clean.csv'
        data = self.pad_lists_to_same_length(self.data)
        df = pd.DataFrame(data)
        df.to_csv(path_save_file, index=False)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = CO2_fitting_app()
    window.show()
    sys.exit(app.exec_())
