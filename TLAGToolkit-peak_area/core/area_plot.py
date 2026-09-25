# -*- coding: utf-8 -*-
"""
ANALYSING ALBA'S DATA IN-SITU TLAG GROWTH'


IMPORTANT: files name must be 'peak_fits_Y11277_REBCO005.csv' or 'peak_fits_Y11277_cooling_dome.csv', but always name of the peak must be last thing before .csv and 
           separated by an _

Created on Wed Mar 13 11:03:30 2024

@author: omola
"""


import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.interpolate import pchip
import colorcet
import math
import os
import csv
import re
import pickle
import statistics
from scipy.interpolate import UnivariateSpline
from pybaselines import spline as sp
from pybaselines.polynomial import poly
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp1d
#from simplified_peak_fitting import fit_peaks


colors_peaks = {
    'CuO': 'green',
    'Cu2O': 'magenta',
    'BaCu2O2': 'gold',
    'BaCO3 ortho': 'red',
    'BaCO3 mono': 'darkred',
    'Cu': 'brown',
    'REBCO(005)': 'blue',
    'REBCO(003)': 'navy',
    'REBCO(103)(110)': 'cornflowerblue',
    'RE2O3': 'pink',
    'Y2Cu2O5': 'magenta',
    'REBCO(004)': 'blueviolet',
    'REBCO(113)': 'teal',
    'REBCO(200)': 'pink',
    'REBCO(102)': 'orchid',
    'BaCO3 hexagonal': 'lime',
    'BaCu3O4': 'turquoise',
    'CuO*': 'maroon',
    'CuO**': 'darkorange',
    'peak5': 'indigo',
    'peak6': 'sienna',
    'peak7': 'deepskyblue',
    'peak9': 'darkolivegreen',
    'peak10': 'cyan',
    'peak12': 'darkcyan',
    'peak15': 'crimson',
    'peak16': 'darkgoldenrod'
}


# FUNCTION DEFINTIONS

# allows multiple selection of csv files
def upload_multiple_csv_files(directory):
    root = tk.Tk()
    root.withdraw()
    csv_files = filedialog.askopenfilenames(                     # peak_fitting_files is a tuple containing the paths to the selected files
                    initialdir = directory,
                    title = 'Select the fitting for all peaks that you want to include on the analysis',
                    filetypes=(('CSV files', '*.csv'), ),
                    multiple = True 
                    )
    
    if not csv_files:
        raise ValueError("No files selected. Please select at least one CSV file.")
    
    return csv_files


def upload_csv_files():
    directory = 'C:/Users/eghiara/Desktop/ALBA/NCD_Mar24/DATA/PROCESSED/'
    peak_fitting_files = upload_multiple_csv_files(directory)

    peak_files = []
    final_file = []
    mass_spectr_file = []
    for file in peak_fitting_files:
        basename = os.path.basename(file)
        if 'final' in basename:
            final_file = file
        elif 'S1_' in basename:
            mass_spectr_file = file
        else:
            peak_files.append(file)

    directory = os.path.dirname(peak_files[0])

    return peak_files, final_file, mass_spectr_file, directory



def get_name(header):
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



def peak_dict(file_path):
    """
    Extracts all peak information from a file and keeps it on a dictionary. if multiple peaks, keep all info together
    """

    peak_parameters = {}
    with open(file_path, newline='') as csvfile:          # open csv file and keep the information of the peak in a dictionary

        header = csvfile.readline().strip().split(',')    # expected to be: imgIndex, temperature, pressure, time, timestamp, + peak parameters
        num_columns = len(header)
        csvdtype = [('imgIndex', 'U4')] + [('col{}'.format(i), float) for i in range(num_columns - 1)]

        data = np.genfromtxt(file_path, delimiter=',', skip_header=1, dtype=csvdtype, filling_values=np.nan, unpack=True)

        # entrances of the dictionary
        peak_parameters = dict(zip(header, data))
        peak_parameters['imgIndex'] = [str(index).zfill(4) for index in peak_parameters['imgIndex']]
        peak_parameters['name'] = get_name(header)

    return peak_parameters



def extract_peak_info(peak_fitting_files):
    """
    If csv file contains more than one peak (multiple peaks on peak_dict), here we separate them for the dictionary 'peaks_data'
    """

    exp_data_desired = ['temperature', 'pressure', 'resistance', 'time', 'timestamp', 'omega', 'att1', 'att2']
    jump_file = next((file for file in peak_fitting_files if 'jump' in file.lower()), None)
    headers = list(peak_dict(jump_file).keys())
    experiment_data = {header: [] for header in exp_data_desired if header in headers}

    peaks_data = {}
    peak_count = 0
    for i, file in enumerate(peak_fitting_files):

        new_peak_info = peak_dict(file)

        # keep only the experiment_data if we do not have this time range of the experiment (round up to 3 decimals)
        t0 = round(new_peak_info['timestamp'][0], 3)
        if t0 not in [round(ts, 3) for ts in experiment_data['timestamp']]:
            for key in experiment_data:
                if key in new_peak_info:
                    experiment_data[key].extend(new_peak_info[key])
            if 'resistance' in experiment_data and 'resistance' not in new_peak_info:
                experiment_data['resistance'].extend(len(new_peak_info[key]) * [np.nan])

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

    return peaks_data, experiment_data


def time_to_seconds(time_str):
    hours, minutes, seconds = time_str.split(':')
    total_seconds = int(hours)*3600 + int(minutes)*60 + float(seconds)
    return round(total_seconds,3)



def read_mass_spectrometer(mass_spectr_file):
    with open(mass_spectr_file, newline='') as csvfile:
        df = pd.read_csv(csvfile)
        mass_spectr_data = df.to_dict(orient='list')

    return mass_spectr_data


# finds the repeated elements on a list and returns the repeated element and its positions
def find_duplicates_with_positions(my_list):
    duplicates = defaultdict(list)
    for i, item in enumerate(my_list):
        duplicates[item].append(i)
    return {item: positions for item, positions in duplicates.items() if len(positions) > 1}



# sort a list and returns both sorted elements and their original indices
def sort_list_keep_index(my_list):
    sorted_elements_with_indices = sorted(enumerate(my_list), key=lambda x: x[1])

    # Extract sorted elements and their original indices
    sorted_elements = [element for index, element in sorted_elements_with_indices]
    original_indices = [index for index, element in sorted_elements_with_indices]

    return sorted_elements, original_indices


def flatten(nested_list):
    flattened_list = [value for sublist in nested_list for value in sublist]
    return flattened_list



def concatenated_dict_data(peaks_data, repeated_peaks_sorted_index):
    keys_to_skip = ['name', 'model']

    first_peak_index = repeated_peaks_sorted_index[0]
    concatenated_peak = peaks_data[f'peak{first_peak_index}']

    for i in repeated_peaks_sorted_index[1:]:  
        peak_name = f'peak{i}'
        for key in peaks_data[peak_name].keys():
            if key not in keys_to_skip:
                try:
                    concatenated_peak[key] = np.concatenate((concatenated_peak[key], peaks_data[peak_name][key]))
                except:
                    print('no key ' + key + ' in function concatenated_dict_data')

    return concatenated_peak


def unify_key_names(peaks):
    for peak in peaks.keys():
        replace_text = peaks[peak]['name'] + '_'
        peaks[peak] = {key.replace(replace_text, ''): value for key, value in peaks[peak].items()}


def join_same_peaks(peaks_data):
    
    list_peaks_name = []
    for i, peak in enumerate(peaks_data):
        list_peaks_name.append(peaks_data[peak]['name'])

    duplicate_peaks_with_positions = find_duplicates_with_positions(list_peaks_name)

    peaks = {}
    if duplicate_peaks_with_positions:

        # for each repeated peak, we order them by time and concatenate all information
        for peak in duplicate_peaks_with_positions.keys():  
            initial_time_list = []
            positions_peak_in_peaks_data = duplicate_peaks_with_positions[peak]
            for position in positions_peak_in_peaks_data:
                initial_time_list.append(peaks_data[f'peak{position}']['timestamp'][0])
            
            initial_time_list_sorted, repeated_peaks_sorted_index = sort_list_keep_index(initial_time_list)
            index_peak = [positions_peak_in_peaks_data[index] for index in repeated_peaks_sorted_index]

            concatenated_peak = concatenated_dict_data(peaks_data, index_peak)
            peak_num = len(peaks)
            peaks[f'peak{peak_num}'] = concatenated_peak

        # copy all not repeated peaks information to the new dictionary such that the new dictionary has the information for all peaks without being repeated
        index_all_repeated_peaks = sorted(flatten(list(duplicate_peaks_with_positions.values())))
        missing_numbers = [num for num in range(len(peaks_data)) if num not in index_all_repeated_peaks]
        for i, peak in enumerate(missing_numbers):
            peak_num = len(peaks)
            peaks[f'peak{peak_num}'] = peaks_data[f'peak{peak}']

    else:
        peaks = peaks_data

    unify_key_names(peaks)     # change name of keys, so it will be easy to call them during plots (change dome_AUC to AUC)
    return peaks


def concatenate_time_values(peaks_data):
    time_values = []
    for peak_key, peak_data in peaks_data.items():
        time_values.extend(peak_data.get('time', []))
    return time_values



def add_normalised_intensity(peaks, time_cooling):
    """
    Add normalised intensity for each peak considering the heating and jump, not cooling
    """
    for peak in peaks.keys():
        try: 
            index_cooling = find_index_of_closest_num(peaks[peak]['time'], time_cooling)
            max_intensity = max(peaks[peak]['I_max'][:index_cooling])
            
            if max_intensity!= 0:
                peaks[peak]['I_max_norm'] = peaks[peak]['I_max'] / max_intensity
                peaks[peak]['I_max_norm_err'] = peaks[peak]['I_max_err'] / max_intensity

                max_AUC = max(peaks[peak]['AUC'][:index_cooling])
                peaks[peak]['AUC_norm'] = peaks[peak]['AUC'] / max_AUC
                peaks[peak]['AUC_norm_err'] = peaks[peak]['AUC_err'] / max_AUC
                
            else:
                max_intensity = max(peaks[peak]['I_max'])
                peaks[peak]['I_max_norm'] = peaks[peak]['I_max'] / max_intensity
                peaks[peak]['I_max_norm_err'] = peaks[peak]['I_max_err'] / max_intensity

                max_AUC = max(peaks[peak]['AUC'][:index_cooling])
                peaks[peak]['AUC_norm'] = peaks[peak]['AUC'] / max_AUC
                peaks[peak]['AUC_norm_err'] = peaks[peak]['AUC_err'] / max_AUC

        except:
            print('Peak ' + peaks[peak]['name'] + ' appears after starting cooling. We normalize it considering the whole range')
            max_intensity = max(peaks[peak]['I_max'])
            peaks[peak]['I_max_norm'] = peaks[peak]['I_max'] / max_intensity
            peaks[peak]['I_max_norm_err'] = peaks[peak]['I_max_err'] / max_intensity

            max_AUC = max(peaks[peak]['AUC'])
            peaks[peak]['AUC_norm'] = peaks[peak]['AUC'] / max_AUC
            peaks[peak]['AUC_norm_err'] = peaks[peak]['AUC_err'] / max_AUC



def sort_dict_by_a_key(data, sort_key):
    zipped_lists = list(zip(*[data[key] for key in data.keys()]))
    sort_index = list(data.keys()).index(sort_key)
    sorted_zipped_lists = sorted(zipped_lists, key=lambda x: x[sort_index])
    sorted_lists = list(zip(*sorted_zipped_lists))

    # Update the original dictionary with sorted lists
    for i, key in enumerate(data.keys()):
        data[key] = list(sorted_lists[i])



def reinicialise_time(peaks, experiment_data):
    sort_dict_by_a_key(experiment_data, 'timestamp')
    t0 = min(experiment_data['timestamp'])
    experiment_data['time'] = experiment_data['timestamp'] - t0
    for peak in peaks.keys():
        peaks[peak]['time'] = peaks[peak]['timestamp'] - t0



def acquisition_time_compensation(peaks, additional_info):
    ''' 
    look at max intensity of the dome of the image when jump occurs and apply this value to all other images to compensate
    for different acqquisition times. We take the jump because at heating and cooling we may take a higher time step, but
    in this area plot we are interested in the jump
    '''

    time_jump = additional_info['time_jump']

    index_dome = find_index_peak(peaks, 'dome')
    index_jump = find_index_of_closest_num(peaks[index_dome]['time'], time_jump)
    intensity_dome_reference = peaks[index_dome]['I_max'][index_jump]
    time_acquisition_at_jump = peaks[index_dome]['time'][index_jump]- peaks[index_dome]['time'][index_jump-1]
    print(f"Impose to all xrd images that I_max(dome) is {intensity_dome_reference:.4f}, which has a time acquisition of {time_acquisition_at_jump:.4f} seconds.")

    additional_info['I_max dome reference'] = intensity_dome_reference
    additional_info['time acquisition reference'] = time_acquisition_at_jump

    # create a dictionary that have at each time (key) the value of the correction to apply (value)
    correction_time_acquisition = []
    for i,intensity_dome in enumerate(peaks[index_dome]['I_max']):
        if intensity_dome == 0:
            imgIndex_value = peaks[index_dome]['imgIndex'][i]
            raise ValueError(f'Intensity of dome is 0 for imgIndex {imgIndex_value:.4f}')
        correction_time_acquisition.append((intensity_dome_reference/intensity_dome, peaks[index_dome]['time'][i]))
    value_dict = {time: value for value, time in correction_time_acquisition}

    for peak in peaks:
        peak_time = peaks[peak]['time']

        for j, time in enumerate(peak_time):
            if time in value_dict:
                peaks[peak]['I_max'][j] = peaks[peak]['I_max'][j] * value_dict[time]
                peaks[peak]['I_max_err'][j] = peaks[peak]['I_max_err'][j] * value_dict[time]
                peaks[peak]['AUC'][j] = peaks[peak]['AUC'][j] * value_dict[time]
                peaks[peak]['AUC_err'][j] = peaks[peak]['AUC_err'][j] * value_dict[time]



def calculate_derivative(x_data, y_data):
    dx = np.diff(x_data)
    dy = np.diff(y_data)
    derivative = dy / dx
    return derivative


def find_pressure_jump(experiment_data, additional_info):
    time = experiment_data['time']
    pressure = experiment_data['pressure']

    pressure_derivative = calculate_derivative(time, pressure)
    index_jump = np.nanargmax(pressure_derivative) 
    # index_jump = 969
    time_jump = time[index_jump]
    pressure_jump = pressure[index_jump]

    additional_info['time_jump'] = time_jump
    additional_info['pressure_jump'] = pressure_jump
    experiment_data['mass_spectrometer'] = []


def remove_background(y_data):
    baseline = sp.irsqr(data=y_data, lam=10, quantile=0.05, num_knots=50, spline_degree=2, diff_order=3, max_iter=500, tol=1e-06, weights=None, eps=None)[0]
    y_without_baseline = y_data - baseline
    return y_without_baseline, baseline

def find_first_index_greater_than_ten(data, threshold):
    for i, value in enumerate(data):
        if value > threshold:
            return i
    return None  # If no such value exists


def baseline_als(y, lam, p, niter=10):
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z
    
    
def match_time_mass_spectrometer(mass_spectr_data, additional_info):
    # in case of clean data, jump is done at last value of mass spectrometer data
    mass_spectr_data['time_clean'] = [x for x in mass_spectr_data['time_clean'] if not math.isnan(x)] 
    shift = mass_spectr_data['time_clean'][-1] - additional_info['time_jump']
    mass_spectr_data['time_clean'] = mass_spectr_data['time_clean'] - shift
    print('hi')
    
    # return index_jump_ms


def find_when_data_starts(mass_spectr_data, additional_info):
    # fit a background on the region of the spectra where we do not have the peaks
    x = mass_spectr_data['time']
    y = np.array(mass_spectr_data['CO2'])
    non_peaks = ( (x < 0) | (x > additional_info['time_jump']+15) )
    x_masked = x[non_peaks]
    y_masked = y[non_peaks]
    
    # _, params = poly(y_masked, x_masked, poly_order=3, return_coef=True)
    # baseline = np.polynomial.Polynomial(params['coef'])(x)
    baseline = baseline_als(y_masked, lam=10**1, p=0.1)    
    baseline_interpolator = interp1d(x_masked, baseline, fill_value="extrapolate")
    baseline = baseline_interpolator(x)
    
    data_without_background = mass_spectr_data['CO2'] - baseline
  
    mean_value = statistics.mean(data_without_background[200:250])
    stand_dev = statistics.stdev(data_without_background[200:250])
    threshold = mean_value + 4*stand_dev 
    index_start_data = find_first_index_greater_than_ten(data_without_background[200:], threshold)
    index_start_data = index_start_data + 200 - 5  
    # index_start_data = 11
    
    fig, ax = plt.subplots(tight_layout={'pad': 0.2})
    ax.plot(mass_spectr_data['time'], mass_spectr_data['CO2'], '-', color='blue', markersize=2, label='mass spectrometer')
    ax.plot(mass_spectr_data['time'], baseline, '--')
    ax.legend(loc='upper left')
    plt.show()
    
    return index_start_data


def adapt_mass_spectr_data_to_plot(mass_spectr_data, experiment_data):
    ''' it adapts the data form mass spectrometer to look nice on the graph. it matches the time with the jump and it changes the y_data to match pressure y axis'''

    CO2_min = min(mass_spectr_data['CO2_clean'][5:])
    mass_spectr_data['CO2_plot'] = [(ms_value-CO2_min) *9e10*2.5 for ms_value in mass_spectr_data['CO2_clean']]
    
    experiment_data['mass_spectrometer'] = mass_spectr_data

   
# given a target number, it finds the index of the closest number on a certain list. this list is meant to be ordered, since the algorithm is optimized for this case
def find_index_of_closest_num(list, target_num):
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

# given a target number, it finds the index of the closest number on a certain list. this list is meant to be ordered and the numbers equidistat, since the 
# algorithm is optimized for this case
def find_index_of_closest_num_equidistant(list, target_num):
    diff = target_num - list[0]
    closest_index = int(diff // (list[1] - list[0]))

    if abs(list[closest_index] - target_num) > abs(list[closest_index + 1] - target_num):
        closest_index += 1

    return closest_index



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


def find_start_cooling(experiment_data, additional_info):
    time = experiment_data['time']
    temperature = experiment_data['temperature']

    index_cooling = find_start_of_increase(temperature)
    time_cooling = time[index_cooling]
    temp_cooling = temperature[index_cooling]

    additional_info['time_cooling'] = time_cooling
    additional_info['temp_cooling'] = temp_cooling



def pad_lists_to_same_length(data):
    max_length = max(len(lst) for lst in data.values())
    for key, lst in data.items():
        if isinstance(lst, np.ndarray):
            lst = lst.tolist()
        if len(lst) < max_length:
            data[key] = lst + [np.nan] * (max_length - len(lst))
    return data


def save_peaks_information(peaks, additional_info, directory):
    path_save_file = directory + '/peaks_data_' + directory[-6:] + '.xlsx'
    with pd.ExcelWriter(path_save_file, engine='openpyxl') as writer:
        for peak, values in peaks.items():
            data = {key: values[key] for key in values if key != 'name'}
            data = pad_lists_to_same_length(data)
            df = pd.DataFrame(data)
            df.to_excel(writer, sheet_name=values['name'], index=False)

        data = {key: additional_info[key] for key in additional_info}
        df = pd.DataFrame([data])
        df.to_excel(writer, sheet_name='additional info', index=False)



def plot_cooling_and_pressure_jump(experiment_data, additional_info):
    time = experiment_data['time']
    temperature = experiment_data['temperature']
    pressure = experiment_data['pressure']
    time_cooling = additional_info['time_cooling']
    temp_cooling = additional_info['temp_cooling']
    time_jump = additional_info['time_jump']
    pressure_jump = additional_info['pressure_jump']

    plt.figure(figsize=(7, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time, temperature, color='blue')
    plt.plot(time_cooling, temp_cooling, 'o', color='red', label='start cooling')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.legend(loc='upper right')

    plt.subplot(2, 1, 2)
    plt.plot(time, pressure, color='orange')
    plt.plot(time_jump, pressure_jump, 'o', color='red', label='start jump')
    plt.xlabel('Time')
    plt.ylabel('Pressure')
    plt.legend(loc='lower right')

    plt.tight_layout()



def find_decreasing_indices(lst):
    ''' it find the indices when the list decreases '''
    decreasing_indices = []
    last_num = lst[0]
    for i in range(len(lst)):
        if lst[i] < last_num:
            decreasing_indices.append(i)
        else:
            last_num = lst[i]
    return decreasing_indices


def remove_elements_at_indices(lst, indices):
    return [value for i, value in enumerate(lst) if i not in indices]


def find_index_peak(peaks_dict, target_peak):
    for peak in peaks_dict:
        name = peaks_dict[peak]['name']
        if name == target_peak:
            index_peak = peak
    return index_peak


def find_omega_correction(final_file_path, REBCO_wrong_omega, dome_wrong_omega, additional_info):
    ''' It returns the ratio between the intensity of REBCO at correct omega and wrong omega after the cooling. We normalized
    both values by its dome in case the time acquisition was different.  '''

    with open(final_file_path, newline='') as csvfile:

        header = csvfile.readline().strip().split(',')
        filtered_header = [col for col in header if 'dome' in col or 'REBCO(005)' in col]
        filtered_header.append('omega')
        indices = [header.index(col) for col in filtered_header]

        data = np.genfromtxt(final_file_path, delimiter=',', skip_header=1, dtype=float, usecols=indices, filling_values=np.nan, unpack=True)
        parameters = dict(zip(filtered_header, data))

        ratio_REBCO_dome_correct_omega = parameters['REBCO(005)_I_max'] / parameters['dome_I_max']
        ratio_REBCO_dome_wrong_omega = REBCO_wrong_omega['I_max'][-1] / dome_wrong_omega['I_max'][-1]
        correction = ratio_REBCO_dome_correct_omega / ratio_REBCO_dome_wrong_omega
        # correction = 2.258

        if  not isinstance(ratio_REBCO_dome_correct_omega, float):
            raise ValueError('final csv file has more than one raw, only one expected')
        
        print(f"Growth at omega {REBCO_wrong_omega['omega'][-1]}.")
        additional_info['growth_omega'] = REBCO_wrong_omega['omega'][-1]
        if parameters['omega'] == 0:
            print('Omega not provided for the final csv file')
        else:
            print(f"Optimal omega for REBCO is {parameters['omega']}.")
            additional_info['optimal_omega'] = parameters['omega']

        additional_info['omega_corr'] = correction
        additional_info['ratio REBCO(005)/dome final'] = ratio_REBCO_dome_correct_omega
        print(f"The correction factor is {correction:.4f}.")
        print(f"The ratio in Imax between the REBCO(005) and the dome at final file is {ratio_REBCO_dome_correct_omega:.4f}.")

        return correction




def correction_wrong_omega(peaks, final_file, additional_info):
    """
    It corrects the intensity and AUC of the REBCO during growth, caused because we were not at the optimal omega during.
    It corrects the REBCO intensity value by multiplying those values by the returns the ratio between the intensity of 
    REBCO at correct omega and wrong omega after the cooling. Correction applied only if the final file exists
    """
    if final_file:
        index_dome = find_index_peak(peaks, 'dome')
        index_REBCO = find_index_peak(peaks, 'REBCO(005)')
        omega_corr = find_omega_correction(final_file, peaks[index_REBCO], peaks[index_dome], additional_info)

        # correct to I_max and AUC of the REBCO and their errors
        peaks[index_REBCO]['I_max'] = [value * omega_corr for value in peaks[index_REBCO]['I_max']]
        peaks[index_REBCO]['AUC'] = [value * omega_corr for value in peaks[index_REBCO]['AUC']]
        peaks[index_REBCO]['I_max_err'] = [value * omega_corr for value in peaks[index_REBCO]['I_max_err']]
        peaks[index_REBCO]['AUC_err'] = [value * omega_corr for value in peaks[index_REBCO]['AUC_err']]

    else:
        print(' ')
        print('WARNING: not final file provided to calculate the ratio between the REBCO(005) and the dome')


def growth_rate(REBCO_005, additional_info, first_index, last_index, smooth, measurement):
    ''' measurement can be: 'intensity' or 'AUC' only '''
    
    if measurement == 'intensity':
        label = 'I_max'
    elif measurement == 'AUC':
        label = 'AUC'
    else:
        raise ValueError(f"Incorrect name of measurement in function growth_rate")
    
    last_zero_index = next((index for index, value in enumerate(REBCO_005[label]) if value != 0), None)-1
    last_index = last_index +50
        
    # growth rate fitting from REBCO(005) peak
    jump_time = np.linspace(REBCO_005['time'][last_zero_index], REBCO_005['time'][last_index], 500)
    # spline_fit = UnivariateSpline(REBCO_005['time'][last_zero_index:-1], REBCO_005[label][last_zero_index:-1], s=smooth)
    # measure_jump = spline_fit(jump_time)
    # coefficients = np.polyfit(REBCO_005['time'][last_zero_index:-1], REBCO_005[label][last_zero_index:-1], 100)
    # poly_fit = np.poly1d(coefficients)
    # measure_jump = poly_fit(jump_time)
    
    # max_value_during_growth = max(measure_jump)
    # growth_rate_REBCO_peak = np.gradient(measure_jump, jump_time) *additional_info['thickness']/max_value_during_growth    
    # additional_info[f"Max growth rate REBCO(005) {measurement}"] = max(growth_rate_REBCO_peak)
    
    # from data
    measurement_data = REBCO_005[label][first_index:last_index]
    time_data = REBCO_005['time'][first_index:last_index]
    max_REBCO_during_growth_data = max(measurement_data)
    growth_rate_REBCO_peak_data = np.gradient(measurement_data, time_data) *additional_info['thickness']/max_REBCO_during_growth_data
    additional_info[f"Max growth rate REBCO(005) {measurement} data"] = max(growth_rate_REBCO_peak_data)
    
    # fig1, ax1 = plt.subplots()
    # ax1.plot(REBCO_005['time'][first_index:last_index], REBCO_005[label][first_index:last_index], '-', color='black')
    # ax1.plot(jump_time, measure_jump, '-', color='red')
    # ax1.grid(True)
    # ax1.set_xlabel('Time (s)')
    # ax1.set_ylabel(f"{measurement} fit")
    
    # fig2, ax2 = plt.subplots()
    # ax2.plot(jump_time, growth_rate_REBCO_peak, '-', color='black')
    # ax2.grid(True)
    # ax2.set_xlabel('Time (s)')
    # ax2.set_ylabel(f"Growth rate REBCO(005) {measurement} (nm/s)")

    fig3, ax3 = plt.subplots()
    ax3.plot(time_data, growth_rate_REBCO_peak_data, '-', color='black')
    ax3.grid(True)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel(f"Growth rate REBCO(005) {measurement} data (nm/s)")



def growth_REBCO005(peaks, additional_info):
    """
    Finds the pressure and temperature at the time of the growth
    Calculated growth rate from conductance and from the integrated intensity of the REBCO(005) peak
    """
    index_REBCO = find_index_peak(peaks, 'REBCO(005)')
    REBCO_005 = peaks[index_REBCO]
    
    # find first index where REBCO(005) is not zero (so, when it starts growing)
    start_growth_REBCO_index = next((index for index, value in enumerate(REBCO_005['I_max']) if value != 0), None)
    additional_info['start_growthREBCO_temp'] = REBCO_005['temperature'][start_growth_REBCO_index]
    additional_info['start_growthREBCO_pressure'] = REBCO_005['pressure'][start_growth_REBCO_index]
    
    # calculate growth rate from the REBCO(005) peak in the interval [-10, 30] seconds of jump
    time_before_jump = additional_info['time_jump'] - 10
    time_after_jump = additional_info['time_jump'] + 30
    
    index_before_jump = find_index_of_closest_num(REBCO_005['time'], time_before_jump)
    index_after_jump = find_index_of_closest_num(REBCO_005['time'], time_after_jump)
    
    # growth rate from REBCO(005) peak integrated intensity and intensity
    growth_rate(REBCO_005, additional_info, index_before_jump, index_after_jump, 0.004, 'AUC')
    growth_rate(REBCO_005, additional_info, index_before_jump, index_after_jump, 0.04, 'intensity')
       
    # growth rate from conductance
    if 'resistance' in REBCO_005:
        conductance = [1/r for r in REBCO_005['resistance']]
        conductance_jump = conductance[index_before_jump:index_after_jump]
        time_jump = REBCO_005['time'][index_before_jump:index_after_jump]
        max_conductance_during_growth = max(conductance_jump)
        growth_rate_conductance = np.gradient(conductance_jump, time_jump) *additional_info['thickness']/max_conductance_during_growth
        additional_info['max_growth_rate_conductance'] = max(growth_rate_conductance)
        
        fig, ax = plt.subplots()
        ax.plot(time_jump, growth_rate_conductance, '-', color='black')
        ax.grid(True)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Growth rate conductance (nm/s)')



def calculate_ratio_random(final_file_path, additional_info):
    ''' It returns the ratio between the intensity of REBCO at correct omega and wrong omega after the cooling. We normalized
    both values by its dome in case the time acquisition was different.  '''

    if final_file_path:
        with open(final_file_path, newline='') as csvfile:

            header = csvfile.readline().strip().split(',')
            filtered_header = [col for col in header if 'REBCO(005)' in col or 'REBCO(103)(110)' in col]
            indices = [header.index(col) for col in filtered_header]

            data = np.genfromtxt(final_file_path, delimiter=',', skip_header=1, dtype=float, usecols=indices, filling_values=np.nan, unpack=True)
            parameters = dict(zip(filtered_header, data))
            if 'REBCO(103)(110)_AUC' in filtered_header:
                ratio = parameters['REBCO(005)_I_max'] / parameters['REBCO(103)(110)_I_max']
                if not isinstance(ratio, float):
                    raise ValueError('final csv file has more than one raw, only one expected')
                
                additional_info['ratio REBCO(005)/REBCO(random) final'] = ratio
                print(f"The ratio in Imax between the REBCO(005) and random at final file is {ratio:.4f}.")



def is_always_increasing(lst):
    # Check if the list is empty or has only one element
    if len(lst) < 2:
        return True
    # Iterate over the list and check if each element is smaller than the next one
    for i in range(1, len(lst)):
        if lst[i] <= lst[i - 1]:
            return False
    return True


def remove_index_on_names(peaks):
    for peak in peaks:
        peak_name = peaks[peak]['name']
        if 'REBCO' not in peak_name:
            peak_name_without_index = re.sub(r'\([\d-]+\)', '', peak_name)
            peaks[peak]['name'] = peak_name_without_index



def plot_2d_figure(peaks, experiment_data, time_start, time_end, color_plot, yaxis, resistance_measurement, directory, save_figure, process):
    time = experiment_data['time']
    temperature = experiment_data['temperature']
    pressure = experiment_data['pressure']

    # specify y axis
    if yaxis == 'I_max_norm':
        yaxis_label = 'Peak max I norm.'
    elif yaxis == 'I_max':
        yaxis_label = 'Peak max I'
    elif yaxis == 'AUC':
        yaxis_label = 'Peak Area'
    elif yaxis == 'AUC_norm':
        yaxis_label = 'Peak Area norm.'

    # plot figure
    fig = plt.figure(figsize=(10, 6))
    plt.subplots_adjust(right=0.7)
    gs = fig.add_gridspec(2, hspace=0, height_ratios=[3, 1])  # Adjust height_ratios to make first plot bigger
    axs = gs.subplots(sharex=True, sharey=False)
    for i,peak in enumerate(peaks.keys()):
        name_peak = peaks[peak]['name']
        if name_peak != 'dome':
            index_start = find_index_of_closest_num(peaks[peak]['time'], time_start)
            index_end = find_index_of_closest_num(peaks[peak]['time'], time_end)

            x_peak = peaks[peak]['time'][index_start:index_end]
            y_peak = np.array(peaks[peak][yaxis][index_start:index_end])
            y_err = np.array(peaks[peak][yaxis+'_err'][index_start:index_end])

            if len(y_peak) != 0:
                if name_peak in colors_peaks:
                    color_plot = colors_peaks[name_peak]
                #to have bar error bars
                #axs[0].errorbar(x_peak, y_peak, yerr=y_err, label=peaks[peak]['name'], color=color_plots[i], fmt='', capsize=5, alpha=0.7)
                #axs[0].plot(x_peak, y_peak, label = peaks[peak]['name'], color=color_plots[i])

                # to have shaded errro bars
                axs[0].plot(x_peak, list(y_peak), label=name_peak, color=color_plot)
                axs[0].fill_between(x_peak, list(y_peak - y_err), list(y_peak + y_err), color=color_plot, alpha=0.5)
                axs[0].fill_between(x_peak, list(y_peak-y_err), color=color_plot, alpha=0.3)

                #to have ONLY the area shaded
                #axs[0].fill_between(x_peak, y_peak, color=color_plots[i], alpha=0.3)
    axs[0].legend(loc='upper left', bbox_to_anchor=(1.15, 1))
    axs[0].set_ylabel(yaxis_label)

    plot_res = 'plot'
    if resistance_measurement in ['resistance', 'conductance']:
        experiment_data['inv_resistance'] = [1/r for r in experiment_data['resistance']]
        index_start = find_index_of_closest_num(experiment_data['time'], time_start)
        index_end = find_index_of_closest_num(experiment_data['time'], time_end)
        ax2r = axs[0].twinx()
        if resistance_measurement == 'resistance':
            ax2r.plot(experiment_data['time'][index_start:index_end], experiment_data['resistance'][index_start:index_end], label='resistance', color='black')
            if process == 'heating':
                ax2r.set_ylim(ymin=8000, ymax=1000000)
            ax2r.set_yscale('log')
            ax2r.set_ylabel('resistance ($\Omega$)')
        else:
            ax2r.plot(experiment_data['time'][index_start:index_end], experiment_data['inv_resistance'][index_start:index_end], label='inv resistance', color='black')
            ax2r.set_ylabel('conductance ($\Omega^{-1}$)')
        
        plot_res = 'resistance_plot'

    # second subplot: temperature and pressure
    index_start2 = find_index_of_closest_num(time, time_start)
    index_end2 = find_index_of_closest_num(time, time_end)
    axs[1].plot(time[index_start2:index_end2], temperature[index_start2:index_end2], label='temperature', color='red')
    ax2 = axs[1].twinx()
    ax2.plot(time[index_start2:index_end2], pressure[index_start2:index_end2], label='pressure', color='blue')
    axs[1].set_ylabel('temperature (ºC)')
    axs[1].set_xlabel('time (s)')
    ax2.set_ylabel('pressure (mbar)')
    
    if experiment_data['mass_spectrometer']:
        mass_spectr_data = experiment_data['mass_spectrometer']
        index_start3 = find_index_of_closest_num(mass_spectr_data['time_clean'], time_start)
        index_end3 = find_index_of_closest_num(mass_spectr_data['time_clean'], time_end)
        ax2.plot(mass_spectr_data['time_clean'][index_start3:index_end3], mass_spectr_data['CO2_plot'][index_start3:index_end3], label='CO2', color='darkorange')
        # ax2.plot(mass_spectr_data['time_exp'][index_start3:index_end3], mass_spectr_data['H2O_plot'][index_start3:index_end3], label='H2O', color='skyblue')
        lines, labels = axs[1].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axs[1].legend(lines + lines2, labels + labels2, loc='upper left')

    # Hide x labels and tick labels for all but bottom plot.
    for ax in axs:
        ax.label_outer()
        
    # Save the figure as png and pickle file
    if save_figure:
        figure_name = directory + '/' + process + f'_{yaxis}_{plot_res}'
        plt.savefig(figure_name + '.png', dpi=500, bbox_inches='tight')
        with open(figure_name + '.pkl', 'wb') as f:
            pickle.dump(fig, f)



def temperature_plot(peaks, experiment_data, time_start, time_end, color_plot, yaxis, resistance_measurement, directory, save_figure, process):

    # specify y axis
    if yaxis == 'I_max_norm':
        yaxis_label = 'Peak max I norm.'
    elif yaxis == 'I_max':
        yaxis_label = 'Peak max I'
    elif yaxis == 'AUC':
        yaxis_label = 'Peak Area'
    elif yaxis == 'AUC_norm':
        yaxis_label = 'Peak Area norm.'

    # plot figure
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(right=0.7)
    for i,peak in enumerate(peaks.keys()):
        name_peak = peaks[peak]['name']
        if name_peak != 'dome':
            index_start = find_index_of_closest_num(peaks[peak]['time'], time_start)
            index_end = find_index_of_closest_num(peaks[peak]['time'], time_end)

            x_peak = peaks[peak]['temperature'][index_start:index_end]
            y_peak = np.array(peaks[peak][yaxis][index_start:index_end])
            y_err = np.array(peaks[peak][yaxis+'_err'][index_start:index_end])

            if len(y_peak) != 0:
                if name_peak in colors_peaks:
                    color_plot = colors_peaks[name_peak]

                # to have shaded errro bars
                ax.plot(x_peak, list(y_peak), label=name_peak, color=color_plot)
                ax.fill_between(x_peak, list(y_peak - y_err), list(y_peak + y_err), color=color_plot, alpha=0.5)
                ax.fill_between(x_peak, list(y_peak-y_err), color=color_plot, alpha=0.3)

    if process == 'cooling':
        ax.invert_xaxis()

    ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1))
    ax.set_ylabel(yaxis_label)
    ax.set_xlabel('temperature (ºC)')

    plot_res = 'plot'
    if resistance_measurement in ['resistance', 'conductance']:
        experiment_data['inv_resistance'] = [1/r for r in experiment_data['resistance']]
        index_start = find_index_of_closest_num(experiment_data['time'], time_start)
        index_end = find_index_of_closest_num(experiment_data['time'], time_end)
        ax2r = ax.twinx()
        
        if resistance_measurement == 'resistance':
            ax2r.plot(experiment_data['temperature'][index_start:index_end], experiment_data['resistance'][index_start:index_end], label='resistance', color='black')
            if process == 'heating':
                ax2r.set_ylim(ymin=8000, ymax=1000000)
            ax2r.grid(visible=True, linestyle=':', color='gray', which='both', alpha=0.7, zorder=-1)
            ax2r.set_yscale('log')
            ax2r.set_ylabel('resistance ($\Omega$)')
        else:
            ax2r.plot(experiment_data['temperature'][index_start:index_end], experiment_data['inv_resistance'][index_start:index_end], label='inv resistance', color='black')
            ax2r.set_ylabel('conductance ($\Omega^{-1}$)')

        
        plot_res = 'resistance_plot'
        
    # Save the figure as png and pickle file
    if save_figure:
        figure_name = directory + '/temp_' + process + f'_{yaxis}_{plot_res}'
        plt.savefig(figure_name + '.png', dpi=500, bbox_inches='tight')
        with open(figure_name + '.pkl', 'wb') as f:
            pickle.dump(fig, f)


def main():
    ''' 
    Three dictionaries gather all data:
        ·peaks: each entrance is a different peak with information 
        ·experiment_data: temperature, pressure, resistance and mass spectrometer data with its own time
        ·additional_info: other information as ratios between REBCO(005) and random and dome, pressure and temperature at jump,
                          omega growth and at jump...
    '''
   
   # PARAMETERS TO CHANGE
   
    time_before_jump = 1000   # 1000 to plot everything
    time_after_jump = 0  # right now the plots use as time_end the time of cooling
    thickness = 400
    process = 'heating'                      # heating, cooling
    resistance_measurement = 'resistance'   # resistance, conductance
    save_figure = False
    save_file = False

    # START OF PROGRAM
    additional_info = {'thickness': thickness}
    files_for_plotting, final_file, mass_spectr_file, directory = upload_csv_files()      # select csv files and keep all peak data in dictionary 'peaks_data'. 
    peaks_data, experiment_data = extract_peak_info(files_for_plotting)                   # each entrance is a peak named 'peak0','peak1',.. with its parameters as specified in 'peak_dict'
    peaks = join_same_peaks(peaks_data)                                                   # if two peaks have the same name (is same peak at different time), we  merge them and order by time

    reinicialise_time(peaks, experiment_data)
    find_pressure_jump(experiment_data, additional_info)

    if mass_spectr_file:
        mass_spectr_data = read_mass_spectrometer(mass_spectr_file)
        match_time_mass_spectrometer(mass_spectr_data, additional_info)
        adapt_mass_spectr_data_to_plot(mass_spectr_data, experiment_data)

    acquisition_time_compensation(peaks, additional_info)
    correction_wrong_omega(peaks, final_file, additional_info)
    find_start_cooling(experiment_data, additional_info)
    add_normalised_intensity(peaks, additional_info['time_cooling'])
    growth_REBCO005(peaks, additional_info)
    calculate_ratio_random(final_file, additional_info)
    remove_index_on_names(peaks)

    if save_file:
        save_peaks_information(peaks, additional_info, directory)


    # MAKE GRAPHS
    time_start = additional_info['time_jump'] - time_before_jump
    # time_end = additional_info['time_jump'] + time_after_jump
    time_end = additional_info['time_cooling']

    plot_cooling_and_pressure_jump(experiment_data, additional_info)
    plot_2d_figure(peaks, experiment_data, time_start, time_end, colors_peaks, 'I_max', False, directory, save_figure, process)
    plot_2d_figure(peaks, experiment_data, time_start, time_end, colors_peaks, 'AUC', False, directory, save_figure, process)
    plot_2d_figure(peaks, experiment_data, time_start, time_end, colors_peaks, 'I_max_norm', False, directory, save_figure, process)
    plot_2d_figure(peaks, experiment_data, time_start, time_end, colors_peaks, 'AUC_norm', False, directory, save_figure, process)

    if 'resistance' in experiment_data:
        plot_2d_figure(peaks, experiment_data, time_start, time_end, colors_peaks, 'I_max', resistance_measurement, directory, save_figure, process)
        plot_2d_figure(peaks, experiment_data, time_start, time_end, colors_peaks, 'AUC', resistance_measurement, directory, save_figure, process)
        plot_2d_figure(peaks, experiment_data, time_start, time_end, colors_peaks, 'I_max_norm', resistance_measurement, directory, save_figure, process)
        plot_2d_figure(peaks, experiment_data, time_start, time_end, colors_peaks, 'AUC_norm', resistance_measurement, directory, save_figure, process)


    if process == 'cooling':
        time_start = additional_info['time_cooling']
    if process == 'heating':
        index_end_heating = find_start_of_decrease(experiment_data['temperature'][150:])
        time_end = experiment_data['time'][index_end_heating+150]
        
    try:
        temperature_plot(peaks, experiment_data, time_start, time_end, colors_peaks, 'I_max', resistance_measurement, directory, save_figure, process)
        temperature_plot(peaks, experiment_data, time_start, time_end, colors_peaks, 'AUC', resistance_measurement, directory, save_figure, process)   
        temperature_plot(peaks, experiment_data, time_start, time_end, colors_peaks, 'I_max_norm', resistance_measurement, directory, save_figure, process)
        temperature_plot(peaks, experiment_data, time_start, time_end, colors_peaks, 'AUC_norm', resistance_measurement, directory, save_figure, process)    
    except:
        print('no temperature plot')
        
        
    plt.show()



if __name__ == "__main__":
    main()




""" # Open file dialog to select a .txt file
log_file = filedialog.askopenfilename(initialdir = 'Z:/PhD/Alba_Feb23/', 
                                      title = 'Select .log file',
                                      filetypes = (('log files', '*.log'),))

# Open file dialog to select a folder containing images
folder_path = filedialog.askdirectory(initialdir = 'Z:/PhD/Alba_Feb23/',
                                      title = 'Select the folder containing the rayonix images')

if not (log_file and folder_path):
    # Raise an exception if not all required data is selected
    raise Exception('You need to select both a .log file and a folder containing the XRD images') """