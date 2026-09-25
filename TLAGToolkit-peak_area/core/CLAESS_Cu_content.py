import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import pchip
import os
import re
import numpy as np

from collections import Counter
from scipy.interpolate import UnivariateSpline, CubicSpline


def find_start_of_increase(data):
    for i in range(len(data) - 1, 0, -1):
        if data[i]  > data[i - 1]:
            return i
    return None  # If no increase is found

def read_csv_files(csv_file):
    with open(csv_file, newline='') as file:
        data = pd.read_csv(file)
    return data


def read_CLAESS_file(file, headers_to_skip):
    with open(file, newline='') as csvfile:        # skip the firsts lines
        for _ in range(headers_to_skip):
            next(csvfile, None)
        lines = csvfile.readlines()
        headers = lines[0].split()
        headers = headers[1:]
        
        data = {header: [] for header in headers}  # keep the headers information
        
        for line in lines[1:]:                     # save data for each header
            row = line.strip().split()
            for i, header in enumerate(headers): 
                data[header].append(float(row[i])) 
    return data


def find_index_of_closest_num(list, target_num):
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


def find_start_heating(lst, num):
    for i in range(len(lst) - num):
        if all(lst[i + j + 1] > lst[i + j] for j in range(num)):
            return i
    return None  # If no such element is found

def calculate_derivative(x_data, y_data):
    dx = np.diff(x_data)
    dy = np.diff(y_data)
    derivative = dy / dx
    return derivative


def rescaling(new_min, new_max, old_min, old_max, list):
    return new_min + (list-old_min)/(old_max-old_min)*(new_max-new_min)





class data_analysis_CLAESS:
    def __init__(self, final_PO2_pressure, channels, fastscan_energy):
        
        self.final_PO2_pressure = final_PO2_pressure
        self.channels = channels
        self.fastscan_energy = fastscan_energy
        self.reference_Cu_norm = 0.5634             # value taken from sample Y13798 from July25
        self.reference_Cu2_norm = 0.1129            # value taken from sample Y13798 from July25
        
        
    # allows multiple selection of csv files
    def upload_multiple_csv_files(self):
        root = tk.Tk()
        root.withdraw()
        files = filedialog.askopenfilenames(
                        # initialdir = directory,
                        title = 'Select the temperature, pressure and timescan files',
                        filetypes = (('CSV and DAT files', '*.csv *.dat'),),
                        multiple = True 
                        )
        
        if not files:
            raise ValueError("No files selected. Please select the temperature, pressure and timescan files.")
        
        self.directory = os.path.dirname(files[0]) 
        return files



    def merge_XAS_data(self, data):
        """ function that merges the XAS data with all the selected channels, and returns this merged normalized by the beam intensity"""
        num_channels = len(self.channels)
        channels_data = {ch: data[ch] for ch in self.channels if ch in data}
        data_merged = [sum(values) / num_channels for values in zip(*channels_data.values())]
        data_merged_and_norm_by_I0 = np.array([x/a for x, a in zip(data_merged, data['a_i0_1'])])
        
        # normalize it by the time adquisition also
        step_time = data['timestamp'][2]- data['timestamp'][1]
        return data_merged_and_norm_by_I0/step_time



    def normalize_XANES(self, energy_data, intensity_data):

        # SUBSTRACT FIRST BACKGROUND AND IMPOSE IT TO BE ZERO
        index1_bcg = 10
        index2_bcg = 50
        slope1, intercept1 = np.polyfit(energy_data[index1_bcg:index2_bcg], intensity_data[index1_bcg:index2_bcg], 1)
        y_background = [slope1 * xi + intercept1 for xi in energy_data]

        plt.figure()
        plt.plot(energy_data, intensity_data)
        plt.plot(energy_data, y_background)
        plt.plot(energy_data[index1_bcg], y_background[index1_bcg], 'o')
        plt.plot(energy_data[index2_bcg], y_background[index2_bcg], 'o')
        plt.xlabel("energy")
        plt.ylabel("Intenisty (a.u.)")
        
        data_without_first_bcg = np.array(intensity_data) - y_background


        # SUBTRACT SECOND BACKGROUND AND IMPOSE IT TO BE 1 
        # energy_start = float(input("Enter the time when data starts (in seconds): "))
        # energy_end = input("Enter the time when data ends (in seconds): ")
        energy_start = 9200
        energy_end = 9800
        index_start = find_index_of_closest_num(energy_data, energy_start)
        index_end = find_index_of_closest_num(energy_data, energy_end)
        
        slope2, intercept2 = np.polyfit(energy_data[index_start:index_end], data_without_first_bcg[index_start:index_end], 1)
        y_background2 = [slope2 * xi + intercept2 for xi in energy_data]

        plt.figure()
        plt.grid()
        plt.plot(energy_data, data_without_first_bcg)
        plt.plot(energy_data, y_background2)
        plt.xlabel("energy")
        plt.ylabel("Intenisty (a.u.)")
        
        # impose that the line in EXAFS is 1
        y_data_normalized = [point / old_max for point, old_max in zip(data_without_first_bcg, y_background2)]       
        
        return y_data_normalized
    
    
    
    def find_normalization_factor(self, XAS_data):
        
        # find, for the selected energy, the value on the XAS spectra normalised and without normalisation       
        norm_energy_fit = pchip(XAS_data['energyc'], XAS_data['normalized intensity'])
        normalized_intensity_value = norm_energy_fit(self.fastscan_energy)

        energy_fit = pchip(XAS_data['energyc'], XAS_data['intensity'])
        intensity_value = energy_fit(self.fastscan_energy)
        
        plt.figure()
        plt.grid()
        plt.plot(XAS_data['energyc'], XAS_data['normalized intensity'], '-')
        plt.plot(self.fastscan_energy, normalized_intensity_value, 'o')
        plt.xlabel("energy")
        plt.ylabel("Intenisty (a.u.)")
        plt.text(
            0.65, 0.15, 
            f"Norm energy: {np.round(normalized_intensity_value,4)} \n" 
            f"Without norm: {np.round(intensity_value,2)}",
            transform=plt.gca().transAxes,  # relative to axes
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
        )
        
        self.normalization_factor = normalized_intensity_value / intensity_value

        
 

    def combine_all_data(self, files):
    
        self.resistance_experiment = False
        timescan_dfs = []
        pressure_dfs = []
        temperature_dfs = []
        XANES_df = []
        
        for file in files:
            if "pressure" in file:
                pressure_data_i = read_csv_files(file)
                pressure_data_i["Pressure_merged"] = [row["Pressure1"] if row["Pressure1"] < 1.3 else row["Pressure2"] for _, row in pressure_data_i.iterrows()]
                pressure_dfs.append(pd.DataFrame(pressure_data_i))
            
            elif "temperature" in file:
                temperature_data_i = read_csv_files(file)
                temperature_dfs.append(pd.DataFrame(temperature_data_i))
                
            elif "resistance" in file:
                self.resistance_experiment = True
                resistance_data = read_csv_files(file)
                
            elif "timescan" in file:
                # find sample name
                parts = file.split('/')
                self.sample_name = parts[-1].split('_')[1]
                
                # keep all data in a df of dfs to later on merge them in a single df
                timescan_data_i = read_CLAESS_file(file, headers_to_skip=5)
                timescan_data_i['merge_intensity'] = self.merge_XAS_data(timescan_data_i)                
                timescan_dfs.append(pd.DataFrame(timescan_data_i))
                
            elif "XAS" in file and file[-4:]==".dat":
                XANES_data_i = read_CLAESS_file(file, headers_to_skip=5)
                XANES_data_i['merge_intensity'] = self.merge_XAS_data(XANES_data_i)
                XANES_df.append(pd.DataFrame(XANES_data_i))
                XAS_file_name = file[:-4]
                
        # sort and merge all timescan, temperature and pressure data
        pressure_data = pd.concat(pressure_dfs, ignore_index=True)
        pressure_data = pressure_data.sort_values(by="Time").reset_index(drop=True)
        temperature_data = pd.concat(temperature_dfs, ignore_index=True)
        temperature_data = temperature_data.sort_values(by="Time").reset_index(drop=True)
        timescan_data = pd.concat(timescan_dfs, ignore_index=True)
        timescan_data = timescan_data.sort_values(by="timestamp").reset_index(drop=True)
                
                
        # average all XANES data to have more statistics, normalize the spectra and save the data in a csv file
        XANES_x_data = np.mean([data['energyc'].values for data in XANES_df], axis=0)
        XANES_y_data = np.mean([data['merge_intensity'].values for data in XANES_df], axis=0)
        XANES_y_data_normalized = self.normalize_XANES(XANES_x_data, XANES_y_data)
        XAS_data = {'energyc': np.array(XANES_x_data), 'intensity': np.array(XANES_y_data), 'normalized intensity': np.array(XANES_y_data_normalized)}
        
        file_name = XAS_file_name[:-8] + '_normalized.csv'
        self.save_dict_to_csv(file_name, XAS_data)
        
        # find the normalization factor for all timescan
        self.find_normalization_factor(XAS_data)
        timescan_data['intensity normalized'] = timescan_data['merge_intensity'] * self.normalization_factor
        
        # rescale the data so that Cu+ and Cu2+ matches the values of the reference sample
        timescan_data['Cu_valence'] = rescaling(2, 1, self.reference_Cu2_norm, self.reference_Cu_norm, timescan_data['intensity normalized'])

                
        # Restart time for all data, t0 being start of heating
        index_start_experiment = find_start_heating(temperature_data["Temperature"], 50)
        t0 = temperature_data["Time"][index_start_experiment]
        pressure_data['Time_relative'] = pressure_data['Time'] - t0
        temperature_data["Time_relative"] = temperature_data["Time"] - t0
        timescan_data["Time_relative"] = list(np.array(timescan_data["timestamp"]) - t0+1.7)     # -t0+83.2 for CLAESS October24
        
        self.time_pressure = pressure_data["Time_relative"]
        self.pressure_data = pressure_data["Pressure_merged"]
        
        temperature_fit = pchip(temperature_data["Time_relative"], temperature_data["Temperature"])
        pressure_fit = pchip(pressure_data["Time_relative"], pressure_data["Pressure_merged"])
        
        if self.resistance_experiment:
            resistance_data["Time_relative"] = resistance_data['Time'] - t0
            resistance_fit = pchip(resistance_data["Time_relative"], resistance_data["Measurement"])
        
        # Interpolate values so we can have combine all data and have them with the time step of fast scan
        times = timescan_data["Time_relative"].values
        timescan_data["temperature"] = np.round(temperature_fit(times), 5)
        timescan_data["total_pressure"] = np.round(pressure_fit(times), 5)
        timescan_data['inv_temperature'] = 1/timescan_data['temperature']
        timescan_data['pressure_PO2'] = timescan_data['total_pressure']/max(timescan_data['total_pressure'])*self.final_PO2_pressure
        
        if self.resistance_experiment:
            timescan_data["resistance"] = np.round(resistance_fit(times), 5)
            timescan_data["conductance"] = 1/timescan_data["resistance"]
        
       
        return timescan_data



    def find_jump(self):      
        pressure_derivative = calculate_derivative(self.all_data['Time_relative'], self.all_data['total_pressure'])
        self.index_jump = np.nanargmax(pressure_derivative) 
        self.time_jump = self.all_data['Time_relative'][self.index_jump]
        self.pressure_at_jump = self.all_data['total_pressure'][self.index_jump]


    def find_start_cooling(self, threshold_temp):
        self.index_cooling = find_start_of_increase(self.all_data['temperature'])
        while self.all_data['temperature'][self.index_cooling] < threshold_temp:
            self.index_cooling = find_start_of_increase(self.all_data['temperature'][:self.index_cooling])
        self.time_cooling= self.all_data['Time_relative'][self.index_cooling]
        self.temp_cooling = self.all_data['temperature'][self.index_cooling]



    def find_Cu_valence_in_YBCO(self):
        # after jump there is Cu2O
        Cu_in_copper_oxide = 1.66/2   # for Cu2O
        self.Cu_valence_YBCO_afterjump = (4.66*self.all_data['Cu_valence'][self.index_cooling]-Cu_in_copper_oxide*2)/3 # for 3:7 solution
        
        # after cooling, around at 700ºC, Cu2O changes to CuO
        index_temp_500 = find_index_of_closest_num(list(self.all_data['temperature'][self.index_cooling:]), 500)
        Cu_in_copper_oxide = 1.66     # for CuO
        self.Cu_valence_YBCO_aftercooling = (4.66*self.all_data['Cu_valence'][index_temp_500+self.index_cooling]-Cu_in_copper_oxide*2)/3 # for 3:7 solution

        print(f"The Cu valence in the YBCO just before cooling is: {round(self.Cu_valence_YBCO_afterjump,3)}")
        print(f"The Cu valence in the YBCO just 1t 500ºC during cooling is: {round(self.Cu_valence_YBCO_aftercooling,3)}")



    def plot_Cu_and_T_PO2(self):
        fig, ax1 = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(right=0.75)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Cu valence', color='tab:blue')
        ax1.plot(self.all_data["Time_relative"], self.all_data["Cu_valence"], color='tab:blue', label='Resistance')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_xlim(left=0)
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Temperature (ºC) ', color='tab:red')
        ax2.plot(self.all_data["Time_relative"], self.all_data["temperature"], color='tab:red', label='Temperature')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.set_ylim(bottom=0)

        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))  # move it outward
        ax3.set_ylabel('Pressure (mbar) ', color='blue')
        ax3.plot(self.time_pressure, self.pressure_data, color='blue')
        ax3.tick_params(axis='y', labelcolor='blue')
        
        plt.savefig(self.directory+'/'+self.sample_name+'_Cu_T_PO2.png', dpi=500, bbox_inches='tight')



    def plot_Cu_and_resistance(self):
        fig, ax1 = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(right=0.75)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Cu valence', color='tab:blue')
        ax1.plot(self.all_data["Time_relative"], self.all_data["Cu_valence"], color='tab:blue', label='Resistance')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_xlim(left=0)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Resistance (Ohms)', color='tab:green')
        ax2.plot(self.all_data["Time_relative"], np.array(self.all_data["resistance"]), color='tab:green', label='Temperature')
        ax2.tick_params(axis='y', labelcolor='tab:green')
        ax2.set_yscale("log")

        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))  # move it outward
        ax3.set_ylabel('Conductance (S)', color='black')
        ax3.plot(self.all_data["Time_relative"], np.array(self.all_data["conductance"]), color='black')
        ax3.tick_params(axis='y', labelcolor='black')
        ax3.set_yscale("log")
        plt.savefig(self.directory+'/'+self.sample_name+'_Cu_resistance.png', dpi=500, bbox_inches='tight')

        fig, ax1 = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(right=0.75)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Cu valence', color='tab:blue')
        ax1.plot(self.all_data["Time_relative"], self.all_data["Cu_valence"], color='tab:blue', label='Resistance')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_xlim(left=0)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Resistance (Ohms)', color='tab:green')
        ax2.plot(self.all_data["Time_relative"], np.array(self.all_data["resistance"]), color='tab:green', label='Temperature')
        ax2.tick_params(axis='y', labelcolor='tab:green')
        ax2.set_yscale("log")
        plt.savefig(self.directory+'/'+self.sample_name+'_Cu_resistance2.png', dpi=500, bbox_inches='tight')
        
        fig, ax1 = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(right=0.75)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Cu valence', color='tab:blue')
        ax1.plot(self.all_data["temperature"], self.all_data["Cu_valence"], color='tab:blue', label='Resistance')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_xlim(left=0)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Resistance (Ohms)', color='tab:green')
        ax2.plot(self.all_data["temperature"], np.array(self.all_data["resistance"]), color='tab:green', label='Temperature')
        ax2.tick_params(axis='y', labelcolor='tab:green')
        ax2.set_yscale("log")
        plt.savefig(self.directory+'/'+self.sample_name+'_Cu_resistance3.png', dpi=500, bbox_inches='tight')


    def plot_timescan(self):
        plt.figure(figsize=(7, 6))
        plt.plot(self.all_data["Time_relative"], self.all_data["Cu_valence"], color='blue')
        plt.xlabel('Time (s)')
        plt.ylabel('Cu valence')
        plt.savefig(self.directory+'/'+self.sample_name+'_timescan2.png', dpi=500, bbox_inches='tight')


    def plot_phase_diagram(self):
        def invT_to_T(x):
            return 1000 / x

        def T_to_invT(x):
            return 1 / (1000*x)
        
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        # sc = ax.scatter(timescan_data['inv_temperature']*1000, timescan_data['pressure_PO2'], c=timescan_data['Cu_valence'], cmap='magma', s=50, vmin=1, vmax=2)
        sc = ax.scatter(self.all_data['inv_temperature']*1000, self.all_data['pressure_PO2'], c=self.all_data['Cu_valence'], cmap='magma', s=50)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Cu valence', fontsize=12)    
        ax.set_xlabel("1/T (1/ºC) [$\cdot 10^{-3}$]", fontsize=14)
        ax.set_ylabel("Pressure (mbar)", fontsize=14)
        ax.set_yscale('log')
        ax.grid(True, which='major', axis='both', color='grey', linestyle='-')
        ax.grid(True, which='minor', axis='both', color='grey', linestyle=':')
        secax = ax.secondary_xaxis('top', functions=(invT_to_T, T_to_invT))
        secax.set_xlabel('Temperature (ºC)')
        
        plt.savefig(self.directory+'_phase_diagram.png', dpi=500, bbox_inches='tight')    
   

    def save_dict_to_csv(self, file_name, data):
        df = pd.DataFrame(data)
        df.to_csv(file_name, index=False)
        print(f'Data save to {file_name}')

        
    def start_analysis(self):
        
        files = self.upload_multiple_csv_files()
        
        self.all_data = self.combine_all_data(files)
        
        self.find_jump()
        self.find_start_cooling(threshold_temp=80)
        
        # self.find_Cu_valence_in_YBCO()
    
        self.plot_Cu_and_T_PO2()            # all plots after analysis
        if self.resistance_experiment:
            self.plot_Cu_and_resistance()
        self.plot_timescan()
        self.plot_phase_diagram()
        
        file_name =self.directory + '/' + self.sample_name + '_merged.csv'
        self.save_dict_to_csv(file_name, self.all_data)
        
    

def main():

    PO2_pressure = 4.2    # rotary oxygen pressure in mbar
    # channels = ['x_ch2_roi1', 'x_ch3_roi1', 'x_ch5_roi1', 'x_ch6_roi1']   # for Oct2024
    channels = ['x_ch2_roi2', 'x_ch3_roi2', 'x_ch5_roi2']   # for June2025, ErBCO
    # channels = ['x_ch2_roi1', 'x_ch3_roi1', 'x_ch5_roi1']   # for June2025, YBCO and GdBCO
    
    # fastscan_energy = 8982    # in eV (for Oct2024)
    fastscan_energy = 8983.33    # in eV (for June2025)
    
    CLAESS_analysis = data_analysis_CLAESS(PO2_pressure, channels, fastscan_energy)
    CLAESS_analysis.start_analysis()
    plt.show()

  


if __name__ == "__main__":
    main()

