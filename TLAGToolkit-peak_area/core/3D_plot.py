# -*- coding: utf-8 -*-
"""
ANALYSING ALBA'S DATA ON IN-SITU TLAG GROWTH'


IMPORTANT: files name must be 'peak_fits_Y11277_YBCO005.csv' or 'peak_fits_Y11277_cooling_dome.csv', but always name of the peak must be last thing before .csv and 
           separated by an _

Created on Wed Mar 13 11:03:30 2024

@author: Ona Mola Bertran
"""



import tkinter as tk
from tkinter import filedialog
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict
import os
from tqdm import tqdm
import copy



# FUNCTION DEFINTIONS

# allows multiple selection of dat files
def upload_multiple_dat_files():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title='Select the folder where the dat files of the data with the background subtraction are')
    
    if not folder_path:
        raise ValueError("No folder selected. Please select the folder where the dat files are.")
    
    # Get all .dat files in the selected folder
    path_data_files = glob.glob(os.path.join(folder_path, '*.dat'))
    path_data_files = [os.path.normpath(path) for path in path_data_files]
    
    if not path_data_files:
        raise ValueError("No .dat files found in the selected folder.")

    return sorted(path_data_files)



def upload_log_file():
    root = tk.Tk()
    root.withdraw()
    log_file = filedialog.askopenfilenames(                     # peak_fitting_files is a tuple containing the paths to the selected files
                    title='Select the log file corresponding to the dat',
                    filetypes=(('log files', '*.log'),),
                    multiple=False 
                    )
    
    if not log_file:
        raise ValueError("No log file selected. Please select the log file.")
    
    return sorted(log_file)


def calculate_derivative(x_data, y_data):
    dx = np.diff(x_data)
    dy = np.diff(y_data)
    derivative = dy / dx
    
    return derivative


def find_pressure_jump(time, pressure):
    pressure_derivative = calculate_derivative(time, pressure)
    index_jump = np.argmax(pressure_derivative)  
    time_jump = time[index_jump]

    return time_jump, index_jump



def extract_data(file_path):

    peak_parameters = {}
    with open(file_path, newline='') as csvfile:          # open csv file and keep the information of the peak in a dictionary

        header = csvfile.readline().strip().split(',')    # expected to be: imgIndex, temperature, pressure, time, + peak parameters
        data = np.genfromtxt(file_path, delimiter=',', skip_header=1, dtype=float, filling_values=np.nan, unpack=True)

        # entrances of the dictionary
        peak_parameters = dict(zip(header, data))

    return peak_parameters



def read_logdata(path_log_file):

    for file in path_log_file:
        with open(file, newline='') as logfile:

            header = logfile.readline().strip().split(',')    # expected to be: imgIndex,scan,time,timestamp,temperature,pressure

            logdtype = [('imgIndex', 'U4'), ('scan','U3'), ('time', float), ('timestamp', float), ('temperature', float), ('pressure', float)]
            data = np.genfromtxt(file, delimiter=',', skip_header=1, dtype=logdtype, filling_values=np.nan, unpack=True)
            logdata = dict(zip(header, data))

    return logdata



def find_index_in_logdata(logdata, target_imgIndex, last_index):

    possible_index = last_index + 1
    possible_imgIndex = logdata['imgIndex'][possible_index]
    if possible_imgIndex == target_imgIndex:
        index = possible_index
        return index
    
    else:
        for index, imgIndex in enumerate(logdata['imgIndex'][possible_index:], start=possible_index):
            if imgIndex == target_imgIndex:
                return index
    
    raise ValueError('No index match found in find_index_in_logdata')



def find_position_imgIndex_in_data_files(path_data_files, imgIndex):

    for index,file in enumerate(path_data_files):
        basename = os.path.basename(file)
        last_part = basename.split('_')[-2:] 
        imgIndex_file = last_part[1][:-4]
        
        if imgIndex_file == imgIndex:
            break

    return index



def find_2theta_range(file, first_2theta, last_2theta):
    """
    For a given file (2theta, intensity) it finds the first and last lines inside the 2theta range
    """

    with open(file, newline='') as datfile:

        data = np.genfromtxt(datfile, names=True)
        data_dict = {key: data[key] for key in data.dtype.names}

        first_index_2t = find_index_of_closest_num_equidistant(data_dict['2th_deg'], first_2theta)
        last_index_2t = find_index_of_closest_num_equidistant(data_dict['2th_deg'], last_2theta)
        lines_skip_end = len(data) - last_index_2t

    return first_index_2t, last_index_2t, lines_skip_end



def join_data(path_data_files, logdata, imgIndex_before_jump, imgIndex_after_jump, first_2theta, last_2theta, step_avoid_data):
    """
    Returns a dictionary 'data_dict' with all data relevant for the plot
    """

    data_dict = {}
    last_index = 0

    first_index_plot = find_position_imgIndex_in_data_files(path_data_files, imgIndex_before_jump)
    last_index_plot = find_position_imgIndex_in_data_files(path_data_files, imgIndex_after_jump)

    first_index_2t, last_index_2t, lines_skip_end = find_2theta_range(path_data_files[0], first_2theta, last_2theta)


    for file in tqdm(path_data_files[first_index_plot-1:last_index_plot:step_avoid_data], desc="keeping data", unit="integrations"):

        basename = os.path.basename(file)
        last_part = basename.split('_')[-2:] 
        log_num = last_part[0]
        imgIndex = last_part[1][:-4]
        key_names = ['2theta', 'intensity']

        name_file = log_num + '_' + imgIndex


        with open(file, newline='') as datfile:

            data = np.genfromtxt(datfile, names=True, skip_header=first_index_2t, skip_footer=lines_skip_end)
            data_dict[name_file] = {key_names[i]: data[key] for i, key in enumerate(data.dtype.names)}
            
            data_dict[name_file]['imgIndex'] = imgIndex
            data_dict[name_file]['scan'] = log_num

            index_log = find_index_in_logdata(logdata, imgIndex, last_index)
            last_index = index_log

            data_dict[name_file]['time'] = logdata['time'][index_log]
            data_dict[name_file]['timestamp'] = logdata['timestamp'][index_log]
            data_dict[name_file]['temperature'] = logdata['temperature'][index_log]
            data_dict[name_file]['pressure'] = logdata['pressure'][index_log]

    return data_dict



def apply_omega_correction(data_dict, YBCO_2theta_start, YBCO_2theta_end, omega_correction, time_jump):

    data_dict_omega_corr = copy.deepcopy(data_dict)

    for file in data_dict:
        for i,angle in enumerate(data_dict[file]['2theta']):
            if time_jump < data_dict[file]['time']:
                if YBCO_2theta_start <= angle <= YBCO_2theta_end:
                    data_dict_omega_corr[file]['intensity'][i] = data_dict[file]['intensity'][i] * omega_correction

    return data_dict_omega_corr



def cut_intensity(data_dict, zmax):

    data_dict_zmax = copy.deepcopy(data_dict)

    for name_file in data_dict:

        # cut graph if intensity>zmax, and in order for the plot to look nice, we set zvalues=zmax of the values that are cutted
        mask_zmax = data_dict_zmax[name_file]['intensity'] <= 1.05*zmax
        transitions = np.argwhere(np.diff(mask_zmax)).flatten()
        odd_indices = transitions[np.arange(len(transitions)) % 2 == 0]
        even_indices = transitions[np.arange(len(transitions)) % 2 != 0]
        data_dict_zmax[name_file]['intensity'][odd_indices] = zmax
        data_dict_zmax[name_file]['intensity'][even_indices+1] = zmax

        # eliminate all values that intensity>zmax (to solve a bug on 3d plot)
        data_dict_zmax[name_file]['intensity'] = data_dict_zmax[name_file]['intensity'][mask_zmax]
        data_dict_zmax[name_file]['2theta'] = data_dict_zmax[name_file]['2theta'][mask_zmax]

    return data_dict_zmax



def find_index_of_closest_num_equidistant(list, target_num):
    """
    Given a target number, it finds the index of the closest number on a certain list.
    Assumes list is in ascending order with equidistant numbers
    """
    diff = target_num - list[0]
    closest_index = int(diff // (list[1] - list[0]))

    if abs(list[closest_index] - target_num) > abs(list[closest_index + 1] - target_num):
        closest_index += 1

    return closest_index



def polygon_under_graph(xlist, ylist):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (xlist, ylist) line graph.  Assumes the xlist are in ascending order.
    """
    return [(xlist[0], 0.), *zip(xlist, ylist), (xlist[-1], 0.)]


# PLOT: INTENSITY OF THE PEAK AS A FUNCTION OF TIME FOR ALL PHASES
def plot_3d_graph(data_dict, time_jump, ylabel):

    if ylabel=='time':
        ylabel_axis = 'Time'

    elif ylabel=='temperature':
        ylabel_axis = 'Temperature'

    elif ylabel=='pressure':
        ylabel_axis = 'Pressure'

    x_list = [v['2theta'] for v in data_dict.values()]
    y_list = [np.tile(v[ylabel], len(v['2theta'])) for v in data_dict.values()]
    z_list = [v['intensity'] for v in data_dict.values()]

    y_plots = [y_list[i][0] for i in range(len(y_list))]

    #cmap = plt.get_cmap('viridis')
    #colors = [cmap(i) for i in np.linspace(0, 1, len(x_list))]

    colors = [(0.3, 0.1, 0.0), (1.0, 0.5, 0.0), (1.0, 0.9, 0.0), (0.0, 0.5, 0.0)]
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
    num_colors = len(x_list)
    colors = [cmap(i / num_colors) for i in range(num_colors)]

    num_ticks = 6  # Adjust the number of ticks as needed
    yticks = np.linspace(min(np.concatenate(y_list)), max(np.concatenate(y_list)), num_ticks, dtype=int)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(projection='3d')

    for x, y, z, c in zip(x_list, y_list, z_list, colors):
        # Plot the line graph given by x and z on the plane y=y with specified color.
        ax.plot(x, y, z, color='black')

    # Make verts a list such that verts[i] is a list of (x, y) pairs defining polygon i.
    verts = [polygon_under_graph(x, y) for x, y in zip(x_list, z_list)]
    poly = PolyCollection(verts, facecolors=colors, alpha=0.9)
    ax.add_collection3d(poly, zs=y_plots, zdir='y')

    # create a red line indicating the jump
    twotheta_line = [x_list[0][0], x_list[0][-1]]
    time_line = [time_jump]*len(twotheta_line)
    intensity_line = [0]*len(twotheta_line)
    ax.plot(twotheta_line, np.array(time_line), np.array(intensity_line), color='red')
    ax.text(twotheta_line[-1], time_jump, 0, ' jump', color='red')

    #ax.set_zlim3d(0,zmax)
    ax.set_box_aspect(aspect = (5,4,3))
    ax.view_init(elev=20., azim=-80)
    ax.set_xlabel('2theta')
    ax.set_ylabel(ylabel_axis)
    ax.set_zlabel('Intensity')

    # On the y-axis let's only label the discrete values that we have data for.
    ax.set_yticks(yticks)

   


def main():

    path_data_files = upload_multiple_dat_files()
    path_log_file = upload_log_file()

    time_before_jump = float(input("Enter the time before jump (in seconds): "))
    time_after_jump = float(input("Enter the time after jump (in seconds): "))
    first_2theta = float(input("Enter the first 2theta (in degrees): "))
    last_2theta = float(input("Enter the last 2theta (in degrees): "))
    zmax = float(input("Enter the zmax (cut on intensity): "))
    step_avoid_data = int(input("Enter the step to avoid data (number of files to skip between plotted data): "))
    omega_correction = float(input("Enter the omega correction: "))
    YBCO_2theta_start = float(input("Enter the YBCO 2theta start (in degrees): "))
    YBCO_2theta_end = float(input("Enter the YBCO 2theta end (in degrees): "))

    logdata = read_logdata(path_log_file)

    # only keep relevant data in 'data_dict', otherwise it takes forever if a lot of files
    time_jump, index_jump = find_pressure_jump(logdata['time'], logdata['pressure'])

    index_before_jump = find_index_of_closest_num_equidistant(logdata['time'], time_jump-time_before_jump)
    index_after_jump = find_index_of_closest_num_equidistant(logdata['time'], time_jump+time_after_jump)

    imgIndex_before_jump = logdata['imgIndex'][index_before_jump]
    imgIndex_after_jump = logdata['imgIndex'][index_after_jump]

    data_dict = join_data(path_data_files, logdata, imgIndex_before_jump, imgIndex_after_jump, first_2theta, last_2theta, step_avoid_data)
    data_dict_omega_corr = apply_omega_correction(data_dict, YBCO_2theta_start, YBCO_2theta_end, omega_correction, time_jump)

    data_dict = cut_intensity(data_dict, zmax)
    data_dict_omega_corr = cut_intensity(data_dict_omega_corr, zmax)

    plot_3d_graph(data_dict, time_jump, 'time')
    plot_3d_graph(data_dict_omega_corr, time_jump, 'time')


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