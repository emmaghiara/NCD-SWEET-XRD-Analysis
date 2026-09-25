
import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.interpolate import pchip
import colorcet
import os
import csv
import re
import pickle
import statistics
from pybaselines import spline as sp
from pybaselines.polynomial import poly
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline, CubicSpline


# FUNCTION DEFINTIONS

# allows multiple selection of csv files
def upload_multiple_csv_files():
    root = tk.Tk()
    root.withdraw()
    csv_files = filedialog.askopenfilenames(
                    # initialdir = directory,
                    title = 'Select the fitting for all peaks that you want to include on the analysis',
                    filetypes = (('csv files', '*.csv'),),
                    multiple = True 
                    )
    
    if not csv_files:
        raise ValueError("No files selected. Please select at least one CSV file.")
    
    directory = os.path.dirname(csv_files[0])
    return csv_files, directory




def time_to_seconds(time_str):
    hours, minutes, seconds = time_str.split(':')
    total_seconds = int(hours)*3600 + int(minutes)*60 + float(seconds)
    return round(total_seconds,3)


def read_mass_spectrometer(mass_spectr_file):
    
    with open(mass_spectr_file, newline='') as csvfile:
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
            data['time'].append(time_to_seconds(row[1]))
            data['H2'].append(float(row[6]))
            data['H2O'].append(float(row[7]))
            data['CO-N2'].append(float(row[8]))
            data['O2'].append(float(row[9]))
            data['Ar'].append(float(row[10]))
            data['CO2'].append(float(row[11]))

    return data

def calculate_derivative(x_data, y_data):
    dx = np.diff(x_data)
    dy = np.diff(y_data)
    derivative = dy / dx
    return derivative


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


def find_jump_in_mass_spectrometer(data):
    # find moment where we close the masss spectrometer, so the jump    
    interpolation = pchip(data['time'],data['CO2'])
    x_interpolation = np.arange(data['time'][0], data['time'][-1], 0.1)
    y_interpolation = interpolation(x_interpolation)
    ms_derivative = calculate_derivative(x_interpolation, y_interpolation)
    jump_ms_time = x_interpolation[np.argmin(ms_derivative)]
    index_jump_ms = find_index_of_closest_num(data['time'], jump_ms_time) - 1
    
    # fig, ax = plt.subplots(tight_layout={'pad': 0.5})
    # ax.plot(x_interpolation, y_interpolation, '-', color='blue', markersize=2, label='mass spectrometer')
    # ax.plot(x_interpolation[1:], ms_derivative, 'o', color='red', label='start jump')
    # ax.legend(loc='upper left')
    # ax.set_ylabel('counts')
    # ax.set_xlabel('time (s)')
    # plt.show()
    
    criteria_for_deciding_that_jump_exists = statistics.stdev(ms_derivative)*8
    abs_value_derivative_jump = abs(min(ms_derivative))
    
    if abs_value_derivative_jump < criteria_for_deciding_that_jump_exists:
        index_jump_ms = None
        
    return index_jump_ms


def find_range_of_data(index_jump, file_data):
    fig, ax = plt.subplots(tight_layout={'pad': 0.5})
    ax.plot(file_data['time'], file_data['CO2'], '-', color='blue', markersize=2, label='mass spectrometer')
    if isinstance(index_jump, int):
        ax.plot(file_data['time'][index_jump], file_data['CO2'][index_jump], 'o', color='red', label='start jump')
    ax.legend(loc='upper left')
    ax.set_ylabel('counts')
    ax.set_xlabel('time (s)')
    fig.suptitle('Decide now which points are to be considered for the background.\n' 
                 'If no background, write 0 and end, and a line will be fitted', fontsize = 12)
    plt.show()
    
    time_start = float(input("Enter the time when data starts (in seconds): "))
    time_end = input("Enter the time when data ends (in seconds): ")
    
    index_start = find_index_of_closest_num(file_data['time'], time_start)
    if time_end == 'end':
        if isinstance(index_jump, int):
            index_end = index_jump
        else:
            index_end = len(file_data['time'])-1
    else:
        time_end = float(time_end)
        index_end = find_index_of_closest_num(file_data['time'], time_end)
    
    if isinstance(index_jump, int):
        if index_end > index_jump:
            raise ValueError("You have to choose a point below the jump.")
    
        x_data = np.array(file_data['time'][0:index_jump+1])
        y_data = np.array(file_data['CO2'][0:index_jump+1])
    else:
        x_data = np.array(file_data['time'])
        y_data = np.array(file_data['CO2'])
            
    return index_start, index_end, x_data, y_data



def substract_background(file_data, index_start, index_end, x_data, y_data, file):
    # fit a straight line fi there is no background before to do a good spline fitting
    if index_start == 0:
        x1 = x_data[index_start]
        y1 = y_data[index_start]
        x2 = x_data[index_end]
        y2 = y_data[index_end]
        
        slope = (y2 - y1) / (x2 - x1)
        indep_par = y1 - slope * x1
        background = slope * x_data + indep_par
    
    # fit a subic spline if there is enough background region
    else:
        background_region = ( ( x_data < x_data[index_start]) | (x_data > x_data[index_end]) )
        x_background = x_data[background_region]
        y_background = y_data[background_region]
        
        print(f"y_background range: {np.min(y_background)} to {np.max(y_background)}")
        
        smooth = 8e-22
        # smooth = (-np.min(y_background)+np.max(y_background))**2/45
        # background = CubicSpline(x_background, y_background)(x_background)
        background = UnivariateSpline(x_background, y_background, k=3, s=smooth)(x_data)
        
    fig, ax = plt.subplots(tight_layout={'pad': 0.5})
    # ax.plot(x_data, y_data, '-', color='blue', markersize=2, label='mass spectrometer')
    ax.plot(file_data['time'], file_data['CO2'], '-', color='blue', markersize=2, label='mass spectrometer')
    ax.plot(x_data, background, '--')
    ax.plot(x_data[index_start], y_data[index_start], 'o', color='orange', label='peaks start')
    ax.plot(x_data[index_end], y_data[index_end], 'o', color='green', label='peaks end')
    ax.legend(loc='upper left')
    ax.set_ylabel('counts')
    ax.set_xlabel('time (s)')
    figure_name = file[:-4] + '_ms1'
    plt.savefig(figure_name + '.png', dpi=500, bbox_inches='tight')

    
    file_data['time_clean'] = x_data
    file_data['CO2_clean'] = y_data - background
    
    fig, ax = plt.subplots(tight_layout={'pad': 0.5})
    ax.plot(file_data['time_clean'], file_data['CO2_clean'], '-', color='blue', markersize=2)
    ax.legend(loc='upper left')
    ax.set_ylabel('counts')
    ax.set_xlabel('time (s)')
    figure_name = file[:-4] + '_ms2'
    plt.savefig(figure_name + '.png', dpi=500, bbox_inches='tight')
    
    
def integrate_CO2_data(mass_spectr_data, file):
    x_data = mass_spectr_data['time_clean']
    y_data = mass_spectr_data['CO2_clean']
    
    y_data_int = []
    accumulative_int = 0
    for i in range(len(x_data) - 2):
        area_i = (x_data[i+1]-x_data[i]) * (y_data[i+1]+y_data[i])/2
        accumulative_int = accumulative_int + area_i
        y_data_int.append(accumulative_int)
    mass_spectr_data['CO2_int'] = y_data_int
        
    x_data_int = []
    for i in range(len(x_data) - 2):  # Loop until the second last element
        x_data_int.append((x_data[i+1]-x_data[i])/2 + x_data[i])
    mass_spectr_data['time_int'] = x_data_int
    
    # plot integrated counts
    fig, ax = plt.subplots(tight_layout={'pad': 0.5})
    ax.plot(mass_spectr_data['time_int'], list(mass_spectr_data['CO2_int']))
    ax.legend(loc='upper left')
    ax.set_ylabel('integrated counts')
    ax.set_xlabel('time (s)')
    figure_name = file[:-4] + '_ms3'
    plt.savefig(figure_name + '.png', dpi=500, bbox_inches='tight')


def find_index_of_closest_num(list, target_num):
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

def pad_lists_to_same_length(data):
    max_length = max(len(lst) for lst in data.values())
    for key, lst in data.items():
        if isinstance(lst, np.ndarray):
            lst = lst.tolist()
        if len(lst) < max_length:
            data[key] = lst + [np.nan] * (max_length - len(lst))
    return data

def save_dict_to_csv(dictionary, directory, name_file):
    path_save_file = directory + '/' + name_file + '_CO2_clean.csv'
    data = pad_lists_to_same_length(dictionary)
    df = pd.DataFrame(data)
    df.to_csv(path_save_file, index=False)
       

def main():
    
    mass_spectr_files, directory = upload_multiple_csv_files()
    
    mass_spectr_data = {}
    for file in mass_spectr_files:
        print(f'Beginning analysis of file {file}')
        file_data = read_mass_spectrometer(file)
        index_jump = find_jump_in_mass_spectrometer(file_data)
        # index_jump = 1000
        index_start, index_end, x_data, y_data = find_range_of_data(index_jump, file_data)
        substract_background(file_data, index_start, index_end, x_data, y_data, file)
        integrate_CO2_data(file_data, file)
        
        # keep information in the dictionary
        start_index = file.find('S1_')
        name_file = file[start_index:-4]
        mass_spectr_data[name_file] = file_data
        
        save_dict_to_csv(file_data, directory, name_file)
        plt.show()



if __name__ == "__main__":
    main()

