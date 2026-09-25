import os
import argparse
import sys
from tkinter import N
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import pchip
from tqdm import tqdm

import yaml

from .utils import BASELINES_FUNCTIONS
from BaselineRemoval import BaselineRemoval

from lmfit import models, Parameter

"""
with open('./Y10444_ybco.yml', 'r') as file:
    prime_service = yaml.safe_load(file)

print(prime_service)
print(prime_service["paths"]["save_path"].format(period = "asdf", scanNo = "asdf"))

"""
def get_yml_content(yml_file):
    with open(yml_file, 'r') as file:
        config_yml = yaml.safe_load(file)

    return config_yml


class Peak_area:
    def __init__(self, config_dict):

        self.config_dict = config_dict

        self.default_baseline = "spline_50_2_lam10"

        self.get_config_dict_variables(self.config_dict)

        self.read_logs()

        self.initialize_intervals()
        

    def get_config_dict_variables(self, config_dict):
        
        self.scan_folder = config_dict['sample']['scan_folder']
        self.facility = config_dict['sample']['facility']
        self.scan_id = str(config_dict['sample']['scan_id'])
        if self.facility == "ALBA":
            self.scan_id = self.scan_id.zfill(3)
        self.period = config_dict['sample']['period']
        self.save_data_baseline_subtraction = config_dict['sample']['save_data_baseline_subtraction']

        self.peak_name = config_dict['xrd']['xrd_name']
        self.index_start = config_dict['xrd'].get('index_start') # Optional
        self.index_end = config_dict['xrd'].get('index_end') # Optional
        self.step = config_dict['xrd'].get('step')
        self.baseline = config_dict['xrd'].get('baseline', self.default_baseline)
        self.peaks_defined = config_dict['xrd']['peaks']
        self.num_peaks = len(self.peaks_defined)

        key_path_integrations = 'path_integrations'
        key_filepath_logs = 'filepath_logs'
        self.path_scans = config_dict['paths'][self.facility][key_path_integrations].format(
            period = self.period, 
            scan_folder = self.scan_folder,
            scan_id = self.scan_id
        )

        self.filepath_logs = config_dict['paths'][self.facility][key_filepath_logs].format(
            period = self.period, 
            scan_folder = self.scan_folder,
            scan_id = self.scan_id
        )

        self.save_path = config_dict['paths'][self.facility]['save_path'].format(
            period = self.period,
            scan_folder = self.scan_folder,
            scan_id = self.scan_id
        )

        self.twoTh_col = {'ALBA': '2th_deg', 'Soleil': '#twoTh'}[self.facility]
        self.intensity_col = {'ALBA': 'I', 'Soleil': 'intensity'}[self.facility]

        self.logs_desired = ["imgIndex", "temperature", "pressure", "resistance",  "time", "timestamp", "omega", "att1", "att2"]

    def read_logs(self):
        try:
            self.df_log = pd.read_csv(self.filepath_logs)
            self.log_columns = self.df_log.columns
        except:
            self.df_log = {"imgIndex": [0], "temperature": [0], "pressure": [0], "resistance": [0], "time": [0], "timestamp": [0], "omega": [0], "att1": [0], "att2": [0]}
            self.df_log = pd.DataFrame(self.df_log)
            self.log_columns = self.df_log.columns

    @staticmethod
    def calculate_auc(x, y, spacing = 0.01):

        data_space = x[1]-x[0]

        if data_space > spacing*2:
            interpolation = pchip(x, y)

            x_auc = np.arange(x[0], x[-1], spacing)
            y_auc = interpolation(x_auc)

            auc = y_auc.sum() * spacing

            # plt.plot(x, y, 'o', label='Data')
            # plt.plot(x_auc, y_auc, 'o', color='red', markersize=2, label='Interpolation')
            # plt.fill_between(x_auc, y_auc, color='skyblue', alpha=0.5)
            # plt.show()

        else:
            auc = y.sum() * data_space

            # plt.plot(x, y, 'o', label='Data')
            # # Plot the rectangular approximation
            # for i in range(len(y)):
            #     plt.bar(x[i], y[i], width=data_space, alpha=0.3, align='center', edgecolor='k')
            # plt.fill_between(x, y, color='skyblue', alpha=0.5)
            # plt.show()


        interpolation = pchip(x, y)
        x_auc = np.arange(x[0], x[-1], spacing)
        y_auc = interpolation(x_auc)
        auc = y_auc.sum() * spacing
        auc_data = y.sum() * data_space

        # plt.plot(x, y, 'o', label='Data')
        # plt.plot(x_auc, y_auc, 'o', color='red', markersize=2, label='Interpolation')
        # plt.fill_between(x_auc, y_auc, color='skyblue', alpha=0.5)
        # # Plot the rectangular approximation
        # for i in range(len(y_auc)):
        #     plt.bar(x_auc[i], y_auc[i], width=spacing, alpha=0.3, align='center', edgecolor='k')
        # plt.show()

        # print('auc = ')
        # print(auc)
        # print('auc data =')
        # print(auc_data)


        return auc_data
    
    
    def find_integration_file(self, path_scans, target_index):
        list_files_integrations = os.listdir(path_scans)

        target_file = None

        for file in list_files_integrations:
            index = self.get_index_integration_file(file)
            if index == target_index:
                target_file = file
                break

        return target_file
    
    
    def peak_calc(self, x, y_corrected, interval, limit, peak_name):
        x_cropped = x[interval[0]:interval[1]]
        y_corrected_cropped = y_corrected[interval[0]:interval[1]]-self.background
        # y_processed_cropped = y_corrected_cropped - min(y_corrected_cropped)
        I_m = max(y_corrected_cropped)
        # background = np.zeros(len(y_corrected))
        # ind = list(range(interval[0], interval[1]))
        # background[ind] += min(y_corrected_cropped)

        # calculate AUC
        if I_m > limit:
            auc= self.calculate_auc(x_cropped, y_corrected_cropped)
            I_max = max(y_corrected_cropped)
            max_index = np.argmax(y_corrected_cropped)
            Twoth_max = x_cropped[max_index]
            I_max_err = self.noise
            auc_err = self.noise * (x_cropped[-1]-x_cropped[0]) / np.sqrt(len(x_cropped))
        else:
            auc = 0
            I_max = 0
            Twoth_max = 0
            I_max_err = 0
            auc_err = 0

        peak_param = {peak_name+'_'+'AUC': auc, peak_name+'_'+'AUC_err': auc_err, peak_name+'_'+'I_max': I_max, peak_name+'_'+'I_max_err': I_max_err, peak_name+'_'+'2th_max': Twoth_max}

        return peak_param
    
    def peaks_area(self):

        # Read the file
        integration_file = self.find_integration_file(self.path_scans, self.index_start)
        # First XRD spectra #000
        df_integration = self.read_integrations_file(os.path.join(self.path_scans, integration_file))
        # 2 theta array is the same in all XRD spectra
        Twotheta = df_integration[self.twoTh_col].values
        self.XRD = pd.DataFrame({'TwoTheta': Twotheta})
        # Noise evaluation
        I = df_integration[self.intensity_col].values
        I_corrected = self.remove_background(I)
        I_noise = I_corrected[2800:2850] # Gradient different 2theta range
        # I_noise = I_corrected[1500:1550]
        self.noise = np.std(I_noise)
        self.background = np.average(I_noise)
        limit = self.background+self.noise*3

        print('The noise (=error) of XRD data scan #0000 is I = {:.3f} a.u.'.format(self.noise))
        print('Noise evaluation between 2 theta = {:.1f}-{:.1f} deg'.format(Twotheta[2800], Twotheta[2850]))
        # print('Noise evaluation between 2 theta = {:.1f}-{:.1f} deg'.format(Twotheta[1500], Twotheta[1550])) # Gradient different 2theta range

        print('The minimum Intensity for signal detection is I = {:.3f} a.u.'.format(self.background+self.noise*2))
        print('The minimum Intensity for signal measure is I = {:.3f} a.u.'.format(limit))

        # Loop through all files in the folder path in numerical order
        self.area_err = {}
        for scan_index in tqdm(range(self.index_start, self.index_end+1, self.step), desc="Peaks evaluation ", unit="integrations"):
            try:
                scan_index_str = str(scan_index).zfill(4)

                # Read the file
                integration_file = self.find_integration_file(self.path_scans, scan_index)
                df_integration = self.read_integrations_file(os.path.join(self.path_scans, integration_file))

                x = df_integration[self.twoTh_col].values
                y = df_integration[self.intensity_col].values

                y_corrected = self.remove_background(y)
                # y_corrected = y_corrected - len(y_corrected)*[self.background]

                result_param = {}
                # Loop through all groups
                for i, peak in enumerate(self.peaks_defined):
                    peak_name = peak['peak']
                    interval = self.intervals[peak_name]
                    #####    2025.04.04 Emma modification for ex situ gradient map #######
                    # modified_interval = tuple(x + 0.0890909*scan_index for x in interval)   # modified to follow the 2theta shift of the peaks because of distance with the detector

                    peak_param = self.peak_calc(x, y_corrected, interval, limit, peak_name)    # change when it is not a map gradient ex-situ
                    #####################
                    result_param.update(peak_param)
                
                for key, value in result_param.items():
                    self.results[key].append(value)

                self.XRD[f'Intensity_{scan_index_str}'] = y_corrected

                # Add file number and logs to the dictionary
                for log in self.logs_recorded:
                    self.results[log].append(
                        self.df_log.loc[self.df_log['imgIndex'] == scan_index, log].values[0]
                    )

            except Exception as e:
                print(e, scan_index)

        self.results = pd.DataFrame(self.results)

        return self.results, self.XRD


    def run_fitting(self):
        self.initialize_dict_results()

        print("Computing the area of selected peaks")
        results, XRD = self.peaks_area()

        self.save_results(results, XRD)

    def save_results(self, results: pd.DataFrame, XRD: pd.DataFrame):

        # creating the corresponding folder
        try:
            os.stat(self.save_path)
        except:
            os.makedirs(self.save_path)

        # Save the Excel
        results.sort_values("imgIndex", inplace=True)

        print("Saving Results in:", self.save_path)
        results.to_csv(
            os.path.join(
                self.save_path, 
                f"PEAKS_results_{self.scan_folder}_{self.peak_name}.csv"
            ),
            index = False
        )

        # Save .dat files with the processed XRD
        # creating the corresponding folder
        if self.save_data_baseline_subtraction:
            self.postprocess_path = os.path.join(self.save_path, f"{self.scan_id}_postprocessed")
            try:
                os.stat(self.postprocess_path)
            except:
                os.makedirs(self.postprocess_path)

            columns_index = self.XRD.columns[1:]
            for i in columns_index:
                i = i[-4:]
                file_name = f'XRD_{self.scan_folder}_{self.scan_id}_{i}.dat'
                columns_in_file = ['TwoTheta', f'Intensity_{i}']
                header = '       2th_deg             I '
                header = '#' + header  # Add # in front of the first header
                with open(os.path.join(self.postprocess_path, file_name),'w') as file:
                    # Write header to the file
                    file.write(header + '\n')
                    self.XRD[columns_in_file].to_csv(file, sep=' ', header=False, index=False, lineterminator='\n')

    def remove_background(self, y):

        # Calculate and remove background
        baseline = BASELINES_FUNCTIONS[self.baseline](data=y)
        y_without_baseline = y - baseline

        return y_without_baseline
    

    def read_initial_integration(self, path_scans):

        initial_integration_file = self.find_integration_file(path_scans, self.index_start)

        df_initial_integration = self.read_integrations_file(os.path.join(self.path_scans, initial_integration_file))

        return df_initial_integration
   
    def read_integrations_file(self, filepath):
        with open(filepath) as f:
            line = f.readline()
            cnt = 0
            while line.startswith('#'):
                prev_line = line
                line = f.readline()
                cnt += 1
                # print(prev_line)

        header = prev_line.strip().lstrip('# ').split()

        df = pd.read_csv(filepath, delimiter="\s+",
                        names=header,
                        skiprows=cnt
                    )

        return df

    def get_index_integration_file(self, file):
        # All facilities share same system
        return int(file.split(".")[0].split("_")[-1])
    
    def initialize_intervals(self):

        # Stablish start index and end index in case they are not specified in the config file
        if self.index_start is None:
            self.index_start = 0
        if self.index_end is None:
            self.index_end = self.df_log["imgIndex"].iloc[-1]

        self.intervals = {}

        initial_integration = self.read_initial_integration(self.path_scans)

        # For each peak determine the 2theta range
        for peak in self.peaks_defined:
            peak_name = peak['peak']
            lowest_2theta = 10**10
            highest_2theta = 0

            # Define 2theta range
            lowest_2theta = min(lowest_2theta, peak["2thlimits"].get("min"))
            highest_2theta = max(highest_2theta, peak["2thlimits"].get("max"))

            lowest_index = self.findClosest(initial_integration[self.twoTh_col], lowest_2theta)
            highest_index = self.findClosest(initial_integration[self.twoTh_col], highest_2theta)
            index_interval = (lowest_index, highest_index)

            # Add everything into the vector
            self.intervals[peak_name] = index_interval
    
    def initialize_dict_results(self):
        self.results = {}

        # Add logs available
        self.logs_recorded = [log_name for log_name in self.logs_desired if log_name in self.log_columns]
        for log in self.logs_recorded:
            self.results[log] = []

        # Add AUC, I_max, Twoth_max param
        for peak in self.peaks_defined:
            self.results[f"{peak['peak']}_AUC"] = []
            self.results[f"{peak['peak']}_AUC_err"] = []
            self.results[f"{peak['peak']}_I_max"] = []
            self.results[f"{peak['peak']}_I_max_err"] = []
            self.results[f"{peak['peak']}_2th_max"] = []

    @staticmethod
    # Returns element closest to target in arr[]
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
                return mid

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
        # return mid and arr[mid]
        return mid

def main():
    # parser = argparse.ArgumentParser(description='Fit peaks of an experiment according to some configuration.')
    # parser.add_argument('yml_file', metavar='yml_file', type=str, nargs=1,
    #                 help='Path of the yaml file with the configuration of the desired peaks to fit')
    # args = parser.parse_args()
    # yml_file = args.yml_file[0]

    yml_file = r'C:\Users\eghiara\Desktop\ALBA\NCD_Mar25\DATA\PROCESSED\Y12645_map_grazing\map.yml'

    config_dict = get_yml_content(yml_file)

    peak_fitter = Peak_area(config_dict)

    peak_fitter.run_fitting()


if __name__ == "__main__":
    main()
