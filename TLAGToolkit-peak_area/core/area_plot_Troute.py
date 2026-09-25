# -*- coding: utf-8 -*-
"""
ANALYSING ALBA'S DATA IN-SITU TLAG GROWTH VIA T-ROUTE

Last version: 28/04/2025

@author: omola
"""


import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import math
import os
import csv
import re
import pickle
import matplotlib.ticker as mticker
#from simplified_peak_fitting import fit_peaks
from scipy.signal import savgol_filter


def find_start_of_increase(data):
    for i in range(len(data) - 1, 0, -1):
        if data[i]  > data[i - 1]:
            return i
    return None  # If no increase is found


def find_start_of_decrease(data):
    for i in range(len(data) - 1):
        if data[i] > data[i + 1]:
            return i
    return None  # If no decrease is found

def find_first_bigger_from_end(lst, x):
    for i in range(len(lst) - 1, -1, -1):  # Iterate backwards
        if lst[i] > x:
            return i  # Return the number and its index
    return None  # If no number is found

def time_to_seconds(time_str):
    hours, minutes, seconds = time_str.split(':')
    total_seconds = int(hours)*3600 + int(minutes)*60 + float(seconds)
    return round(total_seconds,3)
    

def calculate_derivative(x_data, y_data):
    dx = np.diff(x_data)
    dy = np.diff(y_data)
    derivative = dy / dx
    return derivative
    
    
def first_nonzero(lst):
    for i, val in enumerate(lst):
        if val != 0:
            return i  # Return index and value
    return None  # If all elements are zero
    

class data_analysis_NCD:
    def __init__(self, thickness, REBCO_name, process, save_figure, save_file, directory_data, plots, time1_average_grate, time2_average_grate):
        
        self.thickness = thickness
        self.REBCO_name = REBCO_name
        self.process = process
        self.save_figure = save_figure
        self.save_file = save_file
        self.directory_input = directory_data
        self.type_plots = plots
        self.time_line1_average_grate = time1_average_grate
        self.time_line2_average_grate = time2_average_grate
        # self.resistance_measurement = 'conductance'   # resistance, conductance
        
        self.colors_peaks = {
            'CuO': 'gray',
            'Cu2O': 'darkorange',
            'BaCu2O2': 'green',
            'BaCO3 ortho': 'red',
            'BaCO3 mono': 'purple',
            'Cu': 'brown',
            f'{self.REBCO_name}(005)': 'blue',
            f'{self.REBCO_name}(003)': 'sienna',
            f'{self.REBCO_name}(103)': 'olive',
            'Y2Cu2O5': 'gold',
            'Y2BaCuO5': 'cyan',
            f'{self.REBCO_name}(100)': 'magenta',
            f'{self.REBCO_name}(113)': 'teal',
            f'{self.REBCO_name}(200)': 'pink',
            f'{self.REBCO_name}(102)': 'orchid',
            'BaCO3 hexagonal': 'lime',
            'BaCu3O4': 'turquoise',
            'CuO*': 'maroon',
            'CuO**': 'darkred',
            'Y2O3 cubic': 'violet',
            'p': 'cyan',
            'peak5': 'indigo',
            'peak6': 'gray',
            'dd': 'darkolivegreen',
            'epitaxial14': 'darkcyan',
            'epitaxial15': 'crimson',
            'epitaxial16.6': 'deepskyblue',
            'epitaxial17': 'darkgoldenrod',
            'epitaxial17.5': 'darkviolet'
        }
        


    # allows multiple selection of csv files
    def upload_multiple_csv_files(self, directory):
        root = tk.Tk()
        root.withdraw()
        self.csv_files = filedialog.askopenfilenames(                     # peak_fitting_files is a tuple containing the paths to the selected files
                        initialdir = directory,
                        title = 'Select the fitting for all peaks that you want to include on the analysis',
                        filetypes=(('CSV files', '*.csv'), ),
                        multiple = True 
                        )
        
        if not self.csv_files:
            raise ValueError("No files selected. Please select at least one CSV file.")
        
        for file in self.csv_files:
            print(file)


    def upload_csv_files(self):
        self.upload_multiple_csv_files(self.directory_input)

        peak_files = []
        final_file = []
        mass_spectr_file = []
        for file in self.csv_files:
            basename = os.path.basename(file)
            if 'final' in basename:
                final_file = file
            elif 'S1_' in basename:
                mass_spectr_file = file
            else:
                peak_files.append(file)

        self.directory = os.path.dirname(peak_files[0])

        return peak_files, final_file, mass_spectr_file



    def get_name(self, header):
        ''' finds the name of the peak or peaks from the header as the string before '_2th_max' '''

        remove_string = "_2th_max"
        name = []
        for element in header:
            index_imax = element.find(remove_string)
            if index_imax != -1:
                name.append(element[:-len(remove_string)])

        if len(name) == 1:          # if only one peak, return it as a string instead of a list with one element
            name = name[0]
        elif not name:
            raise ValueError("The structure of the headers is not as expected, should have '2th_max' in it.")

        return name



    def peak_dict(self, file_path):
        """ Extracts all peak information from a file and keeps it on a dictionary. if multiple peaks, keep all info together """

        peak_parameters = {}
        with open(file_path, newline='') as csvfile:          # open csv file and keep the information of the peak in a dictionary

            header = csvfile.readline().strip().split(',')    # expected to be: imgIndex, temperature, pressure, time, timestamp, + peak parameters
            csvdtype = [('imgIndex', 'U4')] + [('col{}'.format(i), float) for i in range(len(header) - 1)]
            data = np.genfromtxt(file_path, delimiter=',', skip_header=1, dtype=csvdtype, filling_values=np.nan, unpack=True)

            # entrances of the dictionary
            peak_parameters = dict(zip(header, data))
            peak_parameters['imgIndex'] = [str(index).zfill(4) for index in peak_parameters['imgIndex']]
            peak_parameters['name'] = self.get_name(header)
        return peak_parameters



    def extract_peak_info(self, peak_fitting_files):
        """ If csv file contains more than one peak (multiple peaks on peak_dict), here we separate them for the dictionary 'peaks_data' """

        exp_data_desired = ['temperature', 'pressure', 'resistance', 'time', 'timestamp', 'omega', 'att1', 'att2']
        headers = list(self.peak_dict(peak_fitting_files[0]).keys())
        self.experiment_data = {header: [] for header in exp_data_desired if header in headers}
        
        if 'resistance' in self.experiment_data:
            self.resistance_measurement = True
        else:
            self.resistance_measurement = False

        peaks_data = {}
        peak_count = 0
        for i, file in enumerate(peak_fitting_files):

            new_peak_info = self.peak_dict(file)

            # keep only the experiment_data if we do not have this time range of the experiment (round up to 3 decimals)
            t0 = round(new_peak_info['timestamp'][0], 5)
            if t0 not in [round(ts, 3) for ts in self.experiment_data['timestamp']]:
                for key in self.experiment_data:
                    if key in new_peak_info:
                        self.experiment_data[key].extend(new_peak_info[key])
                if 'resistance' in self.experiment_data and 'resistance' not in new_peak_info:
                    self.experiment_data['resistance'].extend(len(new_peak_info[key]) * [np.nan])

            # check if there is more than one peak information in the csv file
            if isinstance(new_peak_info['name'], list):
                num_peaks_in_dict = len(new_peak_info['name'])
            else:
                num_peaks_in_dict = 1

            # if a file has more than one peak information, here we split this in order to have all peaks in different entrancies
            if num_peaks_in_dict > 1:
                headers_desired = ['imgIndex', 'temperature', 'pressure', 'resistance', 'time', 'timestamp', 'omega', 'att1', 'att2']

                for j, peak in enumerate(new_peak_info['name']):
                    headers = [name for name in list(new_peak_info.keys())]
                    shared_info_headers = [header for header in headers_desired if header in headers]

                    peak_info = [name for name in list(new_peak_info.keys()) if peak in name]   # extract headers with the name of the peak in it
                    header_j = shared_info_headers + peak_info
                    peak_dict_j = {key: new_peak_info[key] for key in header_j}
                    peak_dict_j['name'] = peak

                    peaks_data[f'peak{peak_count}'] = peak_dict_j
                    peak_count = peak_count + 1

            # only one peak on peak_dict
            else:
                peaks_data[f'peak{peak_count}'] = new_peak_info
                peak_count = peak_count + 1

        return peaks_data


    def concatenated_dict_data(self, peaks_data, sorted_indices_for_peaks_repeated):
        keys_to_skip = ['name', 'model']

        first_peak_index = sorted_indices_for_peaks_repeated[0]
        concatenated_peak = peaks_data[f'peak{first_peak_index}']

        for i in sorted_indices_for_peaks_repeated[1:]:  
            peak_name = f'peak{i}'
            for key in peaks_data[peak_name].keys():
                if key not in keys_to_skip:
                    try:
                        concatenated_peak[key] = np.concatenate((concatenated_peak[key], peaks_data[peak_name][key]))
                    except:
                        print('no key ' + key + ' in function concatenated_dict_data')

        return concatenated_peak


    def join_same_peaks(self, peaks_data):
        # finds the repeated elements on a list and returns the repeated element and its positions
        def find_duplicates_with_positions(my_list):
            duplicates = defaultdict(list)
            for i, item in enumerate(my_list):
                duplicates[item].append(i)
            return {item: positions for item, positions in duplicates.items() if len(positions) > 1}
        
        # sort a list and returns their original indices
        def get_sorted_indices(my_list):
            sorted_elements_with_indices = sorted(enumerate(my_list), key=lambda x: x[1])
            return [index for index, element in sorted_elements_with_indices]
        
        def flatten(nested_list):
            return [value for sublist in nested_list for value in sublist]        
        
        list_peaks_name = [peaks_data[peak]['name'] for peak in peaks_data]
        duplicate_peak_indices = find_duplicates_with_positions(list_peaks_name)

        peaks = {}
        if duplicate_peak_indices:

            # for each repeated peak, we order them by time and concatenate all information
            for peak in duplicate_peak_indices.keys():  
                initial_time_list = []
                positions_peak_in_peaks_data = duplicate_peak_indices[peak]
                for position in positions_peak_in_peaks_data:
                    initial_time_list.append(peaks_data[f'peak{position}']['timestamp'][0])
                
                sorted_indices_for_peaks_repeated = get_sorted_indices(initial_time_list)
                index_peak = [positions_peak_in_peaks_data[index] for index in sorted_indices_for_peaks_repeated]

                concatenated_peak = self.concatenated_dict_data(peaks_data, index_peak)
                peak_num = len(peaks)
                peaks[f'peak{peak_num}'] = concatenated_peak

            # copy all not repeated peaks information to the new dictionary such that the new dictionary has the information for all peaks without being repeated
            index_all_repeated_peaks = sorted(flatten(list(duplicate_peak_indices.values())))
            missing_numbers = [num for num in range(len(peaks_data)) if num not in index_all_repeated_peaks]
            for i, peak in enumerate(missing_numbers):
                peak_num = len(peaks)
                peaks[f'peak{peak_num}'] = peaks_data[f'peak{peak}']

        else:
            peaks = peaks_data

        def unify_key_names(peaks):
            for peak in peaks.keys():
                replace_text = peaks[peak]['name'] + '_'
                peaks[peak] = {key.replace(replace_text, ''): value for key, value in peaks[peak].items()}

        unify_key_names(peaks)     # change name of keys, so it will be easy to call them during plots (change dome_AUC to AUC)
        
        return peaks



    def add_normalised_intensity(self):
        """ Add normalised intensity for each peak considering the heating and jump, not cooling """
        
        for peak in self.peaks.keys():
            try: 
                index_cooling = self.find_index_of_closest_num(self.peaks[peak]['time'], self.time_cooling)
                max_intensity = max(self.peaks[peak]['I_max'][:index_cooling])
                max_AUC = max(self.peaks[peak]['AUC'][:index_cooling])              

            except:
                print('Peak ' + self.peaks[peak]['name'] + ' appears after starting cooling. We normalize it considering the whole range')
                max_intensity = max(self.peaks[peak]['I_max'])
                max_AUC = max(self.peaks[peak]['AUC'])
                
            self.peaks[peak]['I_max_norm'] = self.peaks[peak]['I_max'] / max_intensity
            self.peaks[peak]['I_max_norm_err'] = self.peaks[peak]['I_max_err'] / max_intensity
            self.peaks[peak]['AUC_norm'] = self.peaks[peak]['AUC'] / max_AUC
            self.peaks[peak]['AUC_norm_err'] = self.peaks[peak]['AUC_err'] / max_AUC



    def sort_dict_by_a_key(self, data, sort_key):
        zipped_lists = list(zip(*[data[key] for key in data.keys()]))
        sort_index = list(data.keys()).index(sort_key)
        sorted_zipped_lists = sorted(zipped_lists, key=lambda x: x[sort_index])
        sorted_lists = list(zip(*sorted_zipped_lists))

        for i, key in enumerate(data.keys()):        # Update the original dictionary with sorted lists
            data[key] = list(sorted_lists[i])


    def find_start_heating(self, lst, num):
        for i in range(len(lst) - num):
            if all(lst[i + j + 1] > lst[i + j] for j in range(num)):
                return i
        return None  # If no such element is found


    def reinicialise_time(self):
        ''' define start of experiment  when temeprature starts increasing'''
        self.sort_dict_by_a_key(self.experiment_data, 'timestamp')
        self.experiment_data['index_start'] = self.find_start_heating(self.experiment_data['temperature'], 50)
        t0 = self.experiment_data['timestamp'][self.experiment_data['index_start']]+0
        self.experiment_data['time'] = self.experiment_data['timestamp'] - t0
        for peak in self.peaks.keys():
            self.peaks[peak]['time'] = self.peaks[peak]['timestamp'] - t0


    def acquisition_time_compensation(self):
        ''' 
        look at max intensity of the dome of the image when jump occurs and apply this value to all other images to compensate
        for different acqquisition times. We take the jump because at heating and cooling we may take a higher time step, but
        in this area plot we are interested in the jump
        '''

        index_dome = self.find_index_peak(self.peaks, 'dome')
        if self.process == 'growth':
            index_growth = self.find_index_of_closest_num(self.peaks[index_dome]['time'], self.time_growth)
        else:
            index_growth = 10
        index_growth=50
        self.intensity_dome_reference = self.peaks[index_dome]['I_max'][index_growth]
        self.time_acquisition_at_jump = self.peaks[index_dome]['time'][index_growth]- self.peaks[index_dome]['time'][index_growth-1]
        print(f"Impose to all xrd images that I_max(dome) is {self.intensity_dome_reference:.4f}, which has a time acquisition of {self.time_acquisition_at_jump:.4f} seconds.")

        # create a dictionary that have at each time (key) the value of the correction to apply (value)
        correction_time_acquisition = []
        for i,intensity_dome in enumerate(self.peaks[index_dome]['I_max']):
            if intensity_dome == 0:
                imgIndex_value = self.peaks[index_dome]['imgIndex'][i]
                raise ValueError(f'Intensity of dome is 0 for imgIndex {imgIndex_value:.4f}')
            correction_time_acquisition.append((self.intensity_dome_reference/intensity_dome, self.peaks[index_dome]['time'][i]))
        value_dict = {time: value for value, time in correction_time_acquisition}

        for peak in self.peaks:
            peak_time = self.peaks[peak]['time']

            for j, time in enumerate(peak_time):
                if time in value_dict:
                    self.peaks[peak]['I_max'][j] = self.peaks[peak]['I_max'][j] * value_dict[time]
                    self.peaks[peak]['I_max_err'][j] = self.peaks[peak]['I_max_err'][j] * value_dict[time]
                    self.peaks[peak]['AUC'][j] = self.peaks[peak]['AUC'][j] * value_dict[time]
                    self.peaks[peak]['AUC_err'][j] = self.peaks[peak]['AUC_err'][j] * value_dict[time]




    def find_time_growth(self):
        index_REBCO = self.find_index_peak(self.peaks, f'{self.REBCO_name}(005)')
        index_start_growth = first_nonzero(self.peaks[index_REBCO]['I_max'])
        self.time_growth = self.peaks[index_REBCO]['time'][index_start_growth]
        self.temperature_growth = self.peaks[index_REBCO]['temperature'][index_start_growth]
       

    
    def find_index_of_closest_num(self, list, target_num):
        """ given a target number, it finds the index of the closest number on a certain list. this list is meant to be ordered, since the algorithm is optimized for this case """
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


    def find_start_cooling(self, threshold_temp):
        index_cooling = find_start_of_increase(self.experiment_data['temperature'])
        while self.experiment_data['temperature'][index_cooling] < threshold_temp:
            index_cooling = find_start_of_increase(self.experiment_data['temperature'][:index_cooling])
        self.time_cooling = self.experiment_data['time'][index_cooling]
        self.temp_cooling = self.experiment_data['temperature'][index_cooling]


    def pad_lists_to_same_length(self, data):
        max_length = max(len(lst) for lst in data.values())
        for key, lst in data.items():
            if isinstance(lst, np.ndarray):
                lst = lst.tolist()
            if len(lst) < max_length:
                data[key] = lst + [np.nan] * (max_length - len(lst))
        return data


    def save_peaks_information(self):
        path_save_file = self.directory + '/peaks_data_' + self.directory[-6:] + '.xlsx'
        with pd.ExcelWriter(path_save_file, engine='openpyxl') as writer:
            for peak, values in self.peaks.items():
                data = {key: values[key] for key in values if key != 'name'}
                data = self.pad_lists_to_same_length(data)
                df = pd.DataFrame(data)
                df.to_excel(writer, sheet_name=values['name'], index=False)

            # data = {key: additional_info[key] for key in additional_info}
            df = pd.DataFrame([data])
            df.to_excel(writer, sheet_name='additional info', index=False)



    def plot_temperature_pressure(self):
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(self.experiment_data['time'], self.experiment_data['temperature'], color='blue')
        ax.plot(self.time_cooling, self.temp_cooling, 'o', color='red', label='start cooling')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Temperature')
        if self.process == 'growth':
            ax.plot(self.time_growth, self.temperature_growth, 'o', color='blue', label='start growth')
        ax.legend(loc='upper right')
        ax2 = ax.twinx()
        ax2.plot(self.experiment_data['time'], self.experiment_data['pressure'], color='orange')
        ax2.set_ylabel('pressure (mbar)')



    def find_index_peak(self, peaks_dict, target_peak):
        for peak in peaks_dict:
            name = peaks_dict[peak]['name']
            if name == target_peak:
                index_peak = peak
        return index_peak


    def find_omega_correction(self, final_file_path, REBCO_wrong_omega, dome_wrong_omega):
        ''' It returns the ratio between the intensity of REBCO at correct omega and wrong omega after the cooling. We normalized
        both values by its dome in case the time acquisition was different.  '''

        with open(final_file_path, newline='') as csvfile:

            header = csvfile.readline().strip().split(',')
            filtered_header = [col for col in header if 'dome' in col or f'{self.REBCO_name}(005)' in col]
            filtered_header.append('omega')
            indices = [header.index(col) for col in filtered_header]

            data_final_file = np.genfromtxt(final_file_path, delimiter=',', skip_header=1, dtype=float, usecols=indices, filling_values=np.nan, unpack=True)
            XRD_final_file = dict(zip(filtered_header, data_final_file))

            self.ratio_REBCO_dome_correct_omega = XRD_final_file[f'{self.REBCO_name}(005)_I_max'] / XRD_final_file['dome_I_max']
            ratio_REBCO_dome_wrong_omega = REBCO_wrong_omega['I_max'][-1] / dome_wrong_omega['I_max'][-1]
            self.omega_corr = self.ratio_REBCO_dome_correct_omega / ratio_REBCO_dome_wrong_omega

            if not isinstance(self.ratio_REBCO_dome_correct_omega, float):
                raise ValueError('final csv file has more than one raw, only one expected')

            print(f"Growth at omega {REBCO_wrong_omega['omega'][-1]}.")
            self.growth_omega = REBCO_wrong_omega['omega'][-1]
            if XRD_final_file['omega'] == 0:
                self.optimal_omega = "NaN"
                print('Omega not provided for the final csv file')
            else:
                print(f"Optimal omega for REBCO is {XRD_final_file['omega']}.")
                self.optimal_omega = XRD_final_file['omega']

            print(f"The correction factor is {self.omega_corr:.4f}.")
            print(f"The ratio in Imax between the REBCO(005) and the dome at final file is {self.ratio_REBCO_dome_correct_omega:.4f}.")



    def correction_wrong_omega(self, final_file):
        """
        It corrects the intensity and AUC of the REBCO during growth, caused because we were not at the optimal omega during.
        It corrects the REBCO intensity value by multiplying those values by the returns the ratio between the intensity of 
        REBCO at correct omega and wrong omega after the cooling. Correction applied only if the final file exists
        """
        if final_file:
            index_dome = self.find_index_peak(self.peaks, 'dome')
            index_REBCO = self.find_index_peak(self.peaks, f'{self.REBCO_name}(005)')
            self.find_omega_correction(final_file, self.peaks[index_REBCO], self.peaks[index_dome])

            # correct to I_max and AUC of the REBCO and their errors
            for key in ['I_max', 'AUC', 'I_max_err', 'AUC_err']:
                self.peaks[index_REBCO][key] = [value * self.omega_corr for value in self.peaks[index_REBCO][key]]
                
        else:
            print('\n WARNING: not final file provided to calculate the ratio between the REBCO(005) and the dome')


    def instantaneous_growth_rate(self,  measurement, first_index, last_index):
        ''' measurement can be: 'intensity' or 'AUC' only '''
        
        if measurement == 'intensity':
            label = 'I_max'
        elif measurement == 'AUC':
            label = 'AUC'
           
        measurement_data =self.peaks[self.index_REBCO][label][first_index:last_index]
        time_data = self.peaks[self.index_REBCO]['time'][first_index:last_index]
        growth_rate_REBCO_peak_data = np.gradient(measurement_data, time_data) *self.thickness/max(measurement_data)
        max_growth_rate = max(growth_rate_REBCO_peak_data)
        
        index_growth_YBCO = first_nonzero(measurement_data)-1
        measurement_data_filtered = savgol_filter(measurement_data, window_length=11, polyorder=2) 
        growth_rate_REBCO_peak_data_filtered = np.gradient(measurement_data_filtered, time_data) *self.thickness/max(measurement_data_filtered)
        growth_rate_REBCO_peak_data_filtered[1:index_growth_YBCO] = 0

        fig1, ax1 = plt.subplots()
        ax1.plot(time_data, growth_rate_REBCO_peak_data, '-', color='black')
        # ax1.plot(time_data, growth_rate_REBCO_peak_data_filtered, '-', color='blue')
        ax1.grid(True)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel(f"Growth rate {self.REBCO_name}(005) {measurement} data (nm/s)")
        ax1.text(0.95, 0.05, f'max growth rate = {max_growth_rate:.2f} nm/s', fontsize=12, bbox=dict(facecolor='white', edgecolor='black'), 
                transform=ax1.transAxes,  verticalalignment='bottom', horizontalalignment='right')

        return max_growth_rate
        
        
        
    def average_growth_rate(self, measurement, time_line1, time_line2):
        
        if measurement == 'intensity':
            label = 'I_max'
        elif measurement == 'AUC':
            label = 'AUC'
            
        index_REBCO = self.find_index_peak(self.peaks, f'{self.REBCO_name}(005)')
        index_growth = self.find_index_of_closest_num(self.peaks[index_REBCO]['time'], self.time_growth)
        index_cooling = self.find_index_of_closest_num(self.peaks[index_REBCO]['time'], self.time_cooling)
        
        x_axis_plot = self.peaks[index_REBCO]['time'][index_growth-10:index_cooling-150]
        y_axis_plot = self.peaks[index_REBCO][label][index_growth-10:index_cooling-150]
        index_growth_YBCO = first_nonzero(y_axis_plot)-1
        
        index_time_line1 = self.find_index_of_closest_num(x_axis_plot, self.time_growth+time_line1)
        index_time_line2 = self.find_index_of_closest_num(x_axis_plot, self.time_growth+time_line2)
        
        slope1, intercept1 = np.polyfit(x_axis_plot[index_growth_YBCO:index_time_line1], y_axis_plot[index_growth_YBCO:index_time_line1], 1)
        y_fit = [slope1 * xi + intercept1 for xi in x_axis_plot]
        slope2, intercept2 = np.polyfit(x_axis_plot[index_time_line2:-1], y_axis_plot[index_time_line2:-1], 1)
        y_fit2 = [slope2 * xi + intercept2 for xi in x_axis_plot]
        
        time_interception = (intercept2 - intercept1) / (slope1 - slope2)
        y_interception = slope1 * time_interception + intercept1 
        average_growth_rate = self.thickness/(time_interception - x_axis_plot[index_growth_YBCO])
            
        fig, ax = plt.subplots()
        ax.plot(x_axis_plot, y_axis_plot, '-', color='black')
        ax.plot(time_interception, y_interception, 'o', color='red')
        ax.plot(x_axis_plot[index_growth_YBCO], y_axis_plot[index_growth_YBCO], 'o', color='red')
        ax.plot(x_axis_plot, y_fit, color='red')
        ax.plot(x_axis_plot, y_fit2, color='red')
        ax.grid(True)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'{measurement} {self.REBCO_name}(005)')
        ax.set_ylim(min(y_axis_plot), max(y_axis_plot))
        ax.text(0.95, 0.05, f'average growth rate = {average_growth_rate:.2f} nm/s', fontsize=12, bbox=dict(facecolor='white', edgecolor='black'), 
                transform=ax.transAxes,  verticalalignment='bottom', horizontalalignment='right')



    def growth_REBCO005(self, time_line1_average_grate, time_line2_average_grate):
        """
        Finds the pressure and temperature at the time of the growth
        Calculated growth rate from conductance and from the integrated intensity of the REBCO(005) peak
        """
        self.index_REBCO = self.find_index_peak(self.peaks, f'{self.REBCO_name}(005)')
        
        # find first index where REBCO(005) is not zero (so, when it starts growing)
        start_growth_REBCO_index = next((index for index, value in enumerate(self.peaks[self.index_REBCO]['I_max']) if value != 0), None)
        self.temp_REBCO_starts_growing = self.peaks[self.index_REBCO]['temperature'][start_growth_REBCO_index]
        self.pressure_REBCO_starts_growing = self.peaks[self.index_REBCO]['pressure'][start_growth_REBCO_index]

        # calculate growth rate from the REBCO(005) peak in the interval [-10, 80] seconds of jump
        
        index_before_growth = self.find_index_of_closest_num(self.peaks[self.index_REBCO]['time'], self.time_growth - 10)
        index_after_growth = self.find_index_of_closest_num(self.peaks[self.index_REBCO]['time'], self.time_growth + 60)
              
        # growth rate from REBCO(005) peak integrated intensity and intensity
        self.max_growth_rate_AUC = self.instantaneous_growth_rate('AUC', index_before_growth, index_after_growth)
        self.max_growth_rate_intensity = self.instantaneous_growth_rate('intensity', index_before_growth, index_after_growth)
        
        self.average_growth_rate_AUC = self.average_growth_rate('AUC', time_line1_average_grate, time_line2_average_grate)
        self.average_growth_rate_intensity = self.average_growth_rate('intensity', time_line1_average_grate, time_line2_average_grate)
        
        # growth rate from conductance
        if 'resistance' in self.peaks[self.index_REBCO]:
            index_before_growth = self.find_index_of_closest_num(self.experiment_data['time'], self.time_growth - 10)
            index_after_growth = self.find_index_of_closest_num(self.experiment_data['time'], self.time_growth + 60)
        
            self.experiment_data['conductance'] = [1/r for r in self.experiment_data['resistance']]
            max_conductance_during_growth = max(self.experiment_data['conductance'][index_before_growth:index_after_growth])
            self.experiment_data['growth_rate'] = np.gradient(self.experiment_data['conductance'], self.experiment_data['time']) *self.thickness/max_conductance_during_growth
            self.max_growth_rate_conductance = max(self.experiment_data['growth_rate'][index_before_growth:index_after_growth])
            
            fig, ax = plt.subplots()
            ax.plot(self.experiment_data['time'][index_before_growth:index_after_growth], self.experiment_data['growth_rate'][index_before_growth:index_after_growth], '-', color='black')
            ax.grid(True)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Growth rate conductance (nm/s)')
            ax.text(0.95, 0.05, f'max growth rate = {self.max_growth_rate_conductance:.2f} nm/s', fontsize=12, bbox=dict(facecolor='white', edgecolor='black'), 
                    transform=ax.transAxes,  verticalalignment='bottom', horizontalalignment='right')



    def calculate_ratio_random(self, final_file_path):
        ''' It returns the ratio between the intensity of REBCO at correct omega and wrong omega after the cooling. We normalized
        both values by its dome in case the time acquisition was different.  '''

        with open(final_file_path, newline='') as csvfile:

            header = csvfile.readline().strip().split(',')
            filtered_header = [col for col in header if f'{self.REBCO_name}(005)' in col or f'{self.REBCO_name}(103)(110)' in col]
            indices = [header.index(col) for col in filtered_header]

            data = np.genfromtxt(final_file_path, delimiter=',', skip_header=1, dtype=float, usecols=indices, filling_values=np.nan, unpack=True)
            parameters = dict(zip(filtered_header, data))
            if f'{self.REBCO_name}(103)(110)_AUC' in filtered_header:
                self.ratio_epitaxial_random_correct_omega = parameters[f'{self.REBCO_name}(005)_I_max'] / parameters[f'{self.REBCO_name}(103)(110)_I_max']
                if not isinstance(self.ratio_epitaxial_random_correct_omega, float):
                    raise ValueError('final csv file has more than one raw, only one expected')
                
                print(f"The ratio in Imax between the REBCO(005) and random at final file is {self.ratio_epitaxial_random_correct_omega:.4f}.")



    def remove_index_on_names(self):
        for peak in self.peaks:
            peak_name = self.peaks[peak]['name']
            if f'{self.REBCO_name}' not in peak_name:
                peak_name_without_index = re.sub(r'\([\d-]+\)', '', peak_name)
                self.peaks[peak]['name'] = peak_name_without_index



    def plot_2d_figure(self, time_start, time_end, yaxis, resistance_measurement):
        time = self.experiment_data['time']
        temperature = self.experiment_data['temperature']
        pressure = self.experiment_data['pressure']

        # specify y axis
        if yaxis == 'I_max_norm':
            yaxis_label = 'intensity norm'
        elif yaxis == 'I_max':
            yaxis_label = 'intensity'
        elif yaxis == 'AUC':
            yaxis_label = 'peak area'
        elif yaxis == 'AUC_norm':
            yaxis_label = 'peak area norm'

        # plot figure
        fig = plt.figure(figsize=(10, 6))
        plt.subplots_adjust(right=0.7)
        gs = fig.add_gridspec(2, hspace=0, height_ratios=[3, 1])  # Adjust height_ratios to make first plot bigger
        axs = gs.subplots(sharex=True, sharey=False)
        for i,peak in enumerate(self.peaks.keys()):
            name_peak = self.peaks[peak]['name']
            if name_peak != 'dome':
                index_start = self.find_index_of_closest_num(self.peaks[peak]['time'], time_start)
                index_end = self.find_index_of_closest_num(self.peaks[peak]['time'], time_end)

                x_peak = self.peaks[peak]['time'][index_start:index_end]
                y_peak = np.array(self.peaks[peak][yaxis][index_start:index_end])
                y_err = np.array(self.peaks[peak][yaxis+'_err'][index_start:index_end])

                if len(y_peak) != 0:
                    if name_peak in self.colors_peaks:
                        color_plot = self.colors_peaks[name_peak]

                    # to have shaded error bars
                    axs[0].plot(x_peak, list(y_peak), label=name_peak, color=color_plot)
                    axs[0].fill_between(x_peak, list(y_peak - y_err), list(y_peak + y_err), color=color_plot, alpha=0.5)
                    axs[0].fill_between(x_peak, list(y_peak-y_err), color=color_plot, alpha=0.3)

        axs[0].legend(loc='upper left', bbox_to_anchor=(1.15, 1))
        axs[0].set_ylabel(yaxis_label)

        plot_res = 'plot'
        if resistance_measurement in ['resistance', 'conductance']:
            self.experiment_data['inv_resistance'] = [1/r for r in self.experiment_data['resistance']]
            index_start = self.find_index_of_closest_num(self.experiment_data['time'], time_start)
            index_end = self.find_index_of_closest_num(self.experiment_data['time'], time_end)
            ax2r = axs[0].twinx()
            if resistance_measurement == 'resistance':
                ax2r.plot(self.experiment_data['time'][index_start:index_end], self.experiment_data['resistance'][index_start:index_end], label='resistance', color='black')
                if self.process == 'heating':
                    ax2r.set_ylim(ymin=self.ymin_res, ymax=self.ymax_res)
                ax2r.set_yscale('log')
                ax2r.set_ylabel('resistance ($\Omega$)')
            else:
                ax2r.plot(self.experiment_data['time'][index_start:index_end], self.experiment_data['inv_resistance'][index_start:index_end], label='inv resistance', color='black')
                ax2r.set_ylabel('conductance ($\Omega^{-1}$)')
                formatter = mticker.ScalarFormatter(useMathText=True)
                formatter.set_scientific(True)
                formatter.set_powerlimits((-3, 3))
                ax2r.yaxis.set_major_formatter(formatter)
                
            
            plot_res = 'resistance_plot'

        # second subplot: temperature and pressure
        index_start2 = self.find_index_of_closest_num(time, time_start)
        index_end2 = self.find_index_of_closest_num(time, time_end)
        axs[1].plot(time[index_start2:index_end2], temperature[index_start2:index_end2], label='temperature', color='red')
        ax2 = axs[1].twinx()
        ax2.plot(time[index_start2:index_end2], pressure[index_start2:index_end2], label='pressure', color='blue')
        axs[1].set_ylabel('temperature (ºC)')
        axs[1].set_xlabel('time (s)')
        ax2.set_ylabel('pressure (mbar)')
        
        # Hide x labels and tick labels for all but bottom plot.
        for ax in axs:
            ax.label_outer()
            
        # Save the figure as png and pickle file
        if self.save_figure:
            figure_name = self.directory + '/' + self.process + f'_{yaxis}_{plot_res}'
            plt.savefig(figure_name + '.png', dpi=500, bbox_inches='tight')
            with open(figure_name + '.pkl', 'wb') as f:
                pickle.dump(fig, f)


    def temperature_plot(self, time_start, time_end, yaxis, resistance_measurement, process_plot):

        # specify y axis
        if yaxis == 'I_max_norm':
            yaxis_label = 'intensity norm'
        elif yaxis == 'I_max':
            yaxis_label = 'intensity'
        elif yaxis == 'AUC':
            yaxis_label = 'peak area'
        elif yaxis == 'AUC_norm':
            yaxis_label = 'peak area norm'

        # plot figure
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(right=0.7)
        for i,peak in enumerate(self.peaks.keys()):
            name_peak = self.peaks[peak]['name']
            if name_peak != 'dome':
                index_start = self.find_index_of_closest_num(self.peaks[peak]['time'], time_start)
                index_end = self.find_index_of_closest_num(self.peaks[peak]['time'], time_end)

                x_peak = self.peaks[peak]['temperature'][index_start:index_end]
                y_peak = np.array(self.peaks[peak][yaxis][index_start:index_end])
                y_err = np.array(self.peaks[peak][yaxis+'_err'][index_start:index_end])

                if len(y_peak) != 0:
                    if name_peak in self.colors_peaks:
                        color_plot = self.colors_peaks[name_peak]

                    # to have shaded errro bars
                    ax.plot(x_peak, list(y_peak), label=name_peak, color=color_plot)
                    ax.fill_between(x_peak, list(y_peak - y_err), list(y_peak + y_err), color=color_plot, alpha=0.5)
                    ax.fill_between(x_peak, list(y_peak-y_err), color=color_plot, alpha=0.3)

        if process_plot == 'cooling':
            ax.invert_xaxis()

        ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1))
        ax.set_ylabel(yaxis_label)
        ax.set_xlabel('temperature (ºC)')

        plot_res = 'plot'
        if resistance_measurement in ['resistance', 'conductance']:
            self.experiment_data['inv_resistance'] = [1/r for r in self.experiment_data['resistance']]
            index_start = self.find_index_of_closest_num(self.experiment_data['time'], time_start)
            index_end = self.find_index_of_closest_num(self.experiment_data['time'], time_end)
            ax2r = ax.twinx()
            
            if resistance_measurement == 'resistance':
                ax2r.plot(self.experiment_data['temperature'][index_start:index_end], self.experiment_data['resistance'][index_start:index_end], label='resistance', color='black')
                if self.process == 'heating':
                    ax2r.set_ylim(ymin=self.ymin_res, ymax=self.ymax_res)
                ax2r.grid(visible=True, linestyle=':', color='gray', which='both', alpha=0.7, zorder=-1)
                ax2r.set_yscale('log')
                ax2r.set_ylabel('resistance ($\Omega$)')
            else:
                ax2r.plot(self.experiment_data['temperature'][index_start:index_end], self.experiment_data['inv_resistance'][index_start:index_end], label='inv resistance', color='black')
                ax2r.set_ylabel('conductance ($\Omega^{-1}$)')
                formatter = mticker.ScalarFormatter(useMathText=True)
                formatter.set_scientific(True)  # Enable scientific notation when needed
                formatter.set_powerlimits((-3, 3))  # Use scientific notation for numbers outside this range
                ax2r.yaxis.set_major_formatter(formatter)
            
            plot_res = 'resistance_plot'
            
        # Save the figure as png and pickle file
        if self.save_figure:
            figure_name = self.directory + '/temp_' + self.process + f'_{yaxis}_{plot_res}'
            plt.savefig(figure_name + '.png', dpi=500, bbox_inches='tight')
            with open(figure_name + '.pkl', 'wb') as f:
                pickle.dump(fig, f)
                
                
                
    def create_exp_file(self):
        name_file = self.directory[-6:] + '_thickness' + str(self.thickness) + '.txt'
        
        with open(self.directory + '/' + name_file, "w") as file:
            file.write('Time (s), Temperature (ºC), Resistance (Ohms), Conductance (S), Growth rate (nm/s), Pressure (mbar)\n')
            for time, temp, resist, cond, grate, pressure in zip(self.experiment_data['time'], self.experiment_data['temperature'], self.experiment_data['resistance'], self.experiment_data['conductance'], self.experiment_data['growth_rate'], self.experiment_data['pressure']):
                file.write(f"{time:.{6}f}\t{temp:.{4}f}\t{resist:.{6}e}\t{cond:.{6}e}\t{grate:.{6}e}\t{pressure:.{3}f}\n")



    def create_txt_with_additional_info(self):
        name_file = self.directory[-6:] + '_additional_info' + str(self.thickness) + '.txt'
        additional_info = ["thickness", "time_growth", "temperature_growth", "intensity_dome_reference", "time_acquisition_at_jump", "time_cooling", "temp_cooling", 
                           "growth_omega", "optimal_omega", "omega_corr", "ratio_REBCO_dome_correct_omega", "ratio_epitaxial_random_correct_omega", 
                           "temp_REBCO_starts_growing", "pressure_REBCO_starts_growing", "max_growth_rate_AUC", "max_growth_rate_intensity",
                           "average_growth_rate_AUC", "average_growth_rate_intensity"]
        
        existing_info = [var for var in additional_info if hasattr(self, var)]
        
        with open(self.directory + '/' + name_file, "w") as file:
            for var in existing_info:
                file.write(f"{var}={getattr(self, var)}\n")
        
        

    def start_analysis(self):
        ''' 
        Two dictionaries gather all data:
            ·peaks: each entrance is a different peak with information 
            ·experiment_data: temperature, pressure, resistance and mass spectrometer data with its own time
        '''
        
        files_for_plotting, final_file, mass_spectr_file = self.upload_csv_files()      # select csv files and keep all peak data in dictionary 'peaks_data'. 
        self.directory = os.path.dirname(files_for_plotting[0])
        peaks_data = self.extract_peak_info(files_for_plotting)                   # each entrance is a peak named 'peak0','peak1',.. with its parameters as specified in 'peak_dict'
        self.peaks = self.join_same_peaks(peaks_data)                                                   # if two peaks have the same name (is same peak at different time), we  merge them and order by time

        self.reinicialise_time()
        
        if self.process == 'growth':
            self.find_time_growth()
        self.find_start_cooling(threshold_temp=80)
        self.plot_temperature_pressure()
        self.acquisition_time_compensation()
    
        if self.process == 'growth':
            if final_file:
                self.correction_wrong_omega(final_file)
                self.calculate_ratio_random(final_file)
                
            self.add_normalised_intensity()
            try:
                self.growth_REBCO005(self.time_line1_average_grate, self.time_line2_average_grate)
            except:
                print("The growth rate could not be calculated")
            #self.growth_REBCO005(self.time_line1_average_grate, self.time_line2_average_grate)
        self.remove_index_on_names()
        self.create_txt_with_additional_info()


        if self.save_file:
            self.save_peaks_information()

        # MAKE GRAPHS
        time_start = 0
        time_end = self.experiment_data['time'][-1]
            
        if self.resistance_measurement:
            self.create_exp_file()
            if self.thickness == 1200: 
                self.ymin_res = 800
                self.ymax_res = 100000
            else:
                self.ymin_res = 8000
                self.ymax_res = 1000000
            
        # TIME PLOT: ALL PROCESS
        for key in self.type_plots:
            if key in self.peaks['peak0']:
                if 'resistance' in self.experiment_data:
                    for resistance_measurement in ['resistance', 'conductance']:
                        self.plot_2d_figure(time_start, time_end, key, resistance_measurement)
                else:
                    self.plot_2d_figure(time_start, time_end, key, False)


        # TEMPERATURE PLOT: HEATING
        time_end = self.time_cooling + self.experiment_data['index_start']
        for key in self.type_plots:
            if key in self.peaks['peak0']:
                if 'resistance' in self.experiment_data:
                    self.temperature_plot(time_start, time_end, key, resistance_measurement, 'heating')
                else:
                    self.temperature_plot(time_start, time_end, key, False, 'heating')
            
        # TEMPERATURE PLOT: COOLING
        time_start = self.time_cooling
        time_end = self.experiment_data['time'][-1]
        for key in self.type_plots:
            if key in self.peaks['peak0']:
                if 'resistance' in self.experiment_data:
                    self.temperature_plot(time_start, time_end, key, resistance_measurement, 'cooling')
                else:
                    self.temperature_plot(time_start, time_end, key, False, 'cooling')


def main():

    directory_data = 'C:/Users/srasi/Desktop/SILVIA/ALBA/2503_NCD/PROCESSED/'  # you should change it for the directory where your data is
    thickness = 2000            # in nm
    REBCO_name = 'REBCO'        # YBCO, GdBCO,YGdBCO... Whatever name you gave in the excels
    process = 'growth'         # quench, growth
    save_figure = True        # False / True
    save_file = False          # False / True
    plots = ['I_max', 'AUC', 'I_max_norm', 'AUC_norm']  # can be: 'I_max', 'AUC', 'I_max_norm', 'AUC_norm'
    
    # for the average growth rate calculation
    time_line1_average_grate = 60   # (in s) amount of seconds to consider for the first straight line. goes (0, time_line1_average_grate)
    time_line2_average_grate = 40   # (in s) seconds straight line goes from (time_line2_average_grate, cooling)

    XRD_analysis = data_analysis_NCD(thickness, REBCO_name, process, save_figure, save_file, directory_data, plots, time_line1_average_grate, time_line2_average_grate)
    XRD_analysis.start_analysis()
    plt.show()


if __name__ == "__main__":
    main()


