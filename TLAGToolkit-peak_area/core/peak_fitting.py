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


class Peak_fitter:
    def __init__(self, config_dict):

        self.config_dict = config_dict

        self.default_baseline = "imodpoly_2"

        self.get_config_dict_variables(self.config_dict)

        self.define_parameters()

        self.read_logs()

        self.initialize_model()
        

    def get_config_dict_variables(self, config_dict):
        
        self.scan_folder = config_dict['sample']['scan_folder']
        self.facility = config_dict['sample']['facility']
        self.scan_id = str(config_dict['sample']['scan_id'])
        self.fast_scan = config_dict['sample'].get('fast', False)
        if self.facility == "ALBA":
            self.scan_id = self.scan_id.zfill(3)
        self.period = config_dict['sample']['period']

        self.peak_name = config_dict['peaks']['peak_name']
        self.index_start = config_dict['peaks'].get('index_start') # Optional
        self.index_end = config_dict['peaks'].get('index_end') # Optional
        self.step = config_dict['peaks']['step']
        self.baseline = config_dict['peaks'].get('baseline', self.default_baseline)
        self.models_defined = config_dict['peaks']['models']
        self.num_models = len(self.models_defined)

        if self.fast_scan:
            key_path_integrations = 'path_integrations_fast'
            key_filepath_logs = 'filepath_logs_fast'
        else:
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

        
    def define_parameters(self):
        self.x_spacing = 100 # To help improve the convergence of the fitting algorithm. THIS IS ESSENTIAL
        self.peak_interval = 0.15
        self.data_interval = 0
        self.ratio_new_points = 2
        self.min_dist_for_group = 0.2

        self.logs_desired = ["imgIndex", "temperature", "pressure", "time", "resistivity", "timestamp"]

        # Given that we are multiplying 'x' axes by x_spacing, we have to correct the parameters.
        self.correction_factor = {
            'amplitude' : 1/self.x_spacing,
            'center' : 1/self.x_spacing,
            'sigma' : 1/self.x_spacing,
            'gamma' : 1/self.x_spacing,
            'fwhm' : 1/self.x_spacing,
            'height' : 1
        }

    
    def read_logs(self):
        self.df_log = pd.read_csv(self.filepath_logs)

        self.log_columns = self.df_log.columns


    def fit_model(self, model, x_spaced, y_corrected, params, interval, midpoint, ratio_new_points = 1, plot_results = False, i = 0):
        # TODO: End this function. (Test reupdating of central limits)

        y_corrected_cropped = y_corrected[interval[0]:interval[1]]
        y_corrected_cropped = y_corrected_cropped - min(y_corrected_cropped)
        x_spaced_normalized = x_spaced - midpoint
        x_cropped = x_spaced_normalized[interval[0]:interval[1]]

        # Interpolate to have more points
        interpolation = pchip(x_cropped, y_corrected_cropped)

        new_nb_points = int(len(x_cropped) * ratio_new_points)

        new_x = np.linspace(x_cropped[0], x_cropped[-1], new_nb_points)

        new_y = interpolation(new_x)

        result = model.fit(data=new_y, x=new_x, params=params, max_nfev = 5000)

        # Reupdate center limits of each model
        # for key, param in result.params.items():
        #     if key.split("_")[-1] == "center":
        #         result.model.set_param_hint(key, value=param.value, min=param.value - (self.peak_interval*self.x_spacing), max=param.value + (self.peak_interval*self.x_spacing))
        
        if plot_results: #  and i > 100:

            components = result.eval_components(x=new_x)

            for value in components.values():
                plt.plot(new_x, value) # components[model["prefix"] + '_'])
                # pass

            plt.scatter(new_x, new_y, c = "r")
            # timer.start()
            plt.show()

        # calculate AUC
        prefixes = sorted(list(set(param_name.split('_')[0] for param_name in result.params)))
        num_models = len(prefixes)

        # Remove x spacing
        x_original = x_cropped / self.x_spacing

        # Split the interval in num_models
        x_sliced = self.slice_list(x_original, num_models)
        y_sliced = self.slice_list(y_corrected_cropped, num_models)
        auc = {}
        for prefix, x, y in zip(prefixes, x_sliced, y_sliced):
            auc[f"{prefix}_AUC"] = self.calculate_auc(x, y)

        return result, auc

    @staticmethod
    def slice_list(input, size):
        input_size = len(input)
        slice_size = input_size // size
        remain = input_size % size
        result = []
        iterator = iter(input)
        for i in range(size):
            result.append([])
            for _ in range(slice_size):
                result[i].append(next(iterator))
            if remain:
                result[i].append(next(iterator))
                remain -= 1
        return result

    @staticmethod
    def calculate_auc(x, y, spacing = 0.01):
        interpolation = pchip(x, y)

        x_auc = np.arange(x[0], x[-1], spacing)
        y_auc = interpolation(x_auc)

        auc = y_auc.sum() * spacing

        return auc

    def continuous_fitting(self, fitting_direction="forward", plot_mse=False):
        mse_integrations = []

        if fitting_direction == "forward":
            initial_point = self.index_start
            end_point = self.index_end
            step = self.step

        elif fitting_direction == "backward":
            initial_point = self.index_end
            end_point = self.index_start
            step = -self.step
        # Loop through all files in the folder path in numerical order
        for scan_index in tqdm(range(initial_point, end_point, step), desc="Fitting: ", unit="integrations"):
        # for scan_index in tqdm(range(self.index_start, self.index_end, self.step), desc="Fitting: ", unit="integrations"):
            try:
                scan_index_str = str(scan_index).zfill(4)

                # Read the file
                integration_file = self.find_integration_file(self.path_scans, scan_index)
                df_integration = self.read_integrations_file(os.path.join(self.path_scans, integration_file), self.facility)

                x = df_integration[self.twoTh_col].values
                y = df_integration[self.intensity_col].values

                x_spaced = x * self.x_spacing
                # TODO: Improve remove_background function
                y_corrected = self.remove_background(y)

                mse_models = [None for i in range(len(self.group2model))]
                result_params_models = None

                # Loop through all groups
                for i, group in enumerate(self.group2model.keys()):
                    model = self.models[group]
                    params = self.params[group]
                    interval = self.intervals[group]
                    midpoint = self.midpoints[group] * self.x_spacing

                    fitted_model, auc = self.fit_model(model, x_spaced, y_corrected, params, interval, midpoint, ratio_new_points=self.ratio_new_points, plot_results=False, i=(scan_index - self.index_start)//5)

                    mse_models[i] = np.mean(np.power(fitted_model.residual, 2))

                    # Save current parameters for next fitting. This IMPROVES a lot the fitting and avoids artifacts.
                    current_params = fitted_model.params
                    self.params[group] = current_params

                    if i == 0:
                        pass
                        # print("center", current_params["center"])
                        # print(current_params)

                    # Insert AUC as a param
                    for auc_prefix, auc_value in auc.items():
                        current_params[auc_prefix] = Parameter(name=auc_prefix, value=auc_value)

                    if result_params_models is None:
                        result_params_models = current_params
                    else:
                        result_params_models += current_params

                # Add current mse to mse list
                mse_integrations.append(mse_models)

                # Add file number and logs to the dictionary
                for log in self.logs_recorded:
                    self.fitting_data[log].append(
                        self.df_log.loc[self.df_log['imgIndex'] == scan_index, log].values[0]
                    )

                # Add params of the model of this file to the dictionary
                for param_name, param in result_params_models.items():
                    group, function_parameter = param_name.split('_')
                    correction = self.correction_factor.get(function_parameter, 1)
                    if param_name.split('_')[1] == 'center':
                        self.fitting_data[param_name].append((param.value * correction) + self.midpoints[group])
                    else:
                        self.fitting_data[param_name].append(param.value * correction)

            except Exception as e:
                print(e, scan_index)

        self.mse_integrations_mean = np.mean(mse_integrations, axis = 0)
        mse_integrations_transposed = np.transpose(mse_integrations)

        if plot_mse:
            for model_mse in mse_integrations_transposed:
                plt.plot(model_mse)

            plt.show()

        return self.mse_integrations_mean, self.fitting_data

    def run_fitting(self):
        # TODO: Implement AUC
        self.initialize_fitting_data()

        print("Trying to fit your models forward (indices increasing):")
        mse_forward, final_parameters_forward = self.continuous_fitting(fitting_direction="forward")

        print("Trying to fit your models backward (indices decreasing):")
        mse_backward, final_parameters_backward = self.continuous_fitting(fitting_direction="backward")

        print("Mean Squared Error (MSE) of forward fitting:", mse_forward)
        print("Mean Squared Error (MSE) of backward fitting:", mse_backward)
        if (mse_forward < mse_backward).all():
            print("Forward fitting is better. Saving it:")
            self.save_results(final_parameters_forward)
        elif (mse_forward > mse_backward).all():
            print("Backward fitting is better. Saving it.")
            self.save_results(final_parameters_backward)
        else:
            print(f"WARNING: No clear improvements among the {len(mse_forward)} grouped models with forward or backward.\n" \
                  f"The fitting could be improved by fitting your {len(mse_forward)} grouped models separately.")
            self.save_results(final_parameters_backward)

    def save_results(self, fitting_data):

        # creating the corresponding folder
        try:
            os.stat(self.save_path)
        except:
            os.makedirs(self.save_path)

        # Save the fitted model
        model_data_df = pd.DataFrame(fitting_data)
        model_data_df.sort_values("imgIndex", inplace=True)

        print("Saving model in:", self.save_path)
        model_data_df.to_csv(
            os.path.join(
                self.save_path, 
                f"TEST_peak_fits_{self.scan_folder}_{self.peak_name}.csv"
            ),
            index = False
        )

        # Save the yml file in the same location
        self.config_dict["errors"] =  [{model['prefix'] : float(error)} for model, error in zip(self.models_defined, self.mse_integrations_mean)]
        with open(os.path.join(
                self.save_path, 
                f"config_peak_fits_{self.scan_folder}_{self.peak_name}.yml"
                ), "w" 
            ) as file:
            yaml.dump(self.config_dict, file)

    def remove_background(self, y):

        # Calculate and remove background
        baseline = BASELINES_FUNCTIONS[self.baseline](data=y)
        y_without_baseline = y - baseline

        return y_without_baseline


    def initialize_fitting_data(self):
        self.fitting_data = {}

        # Add logs available
        self.logs_recorded = [log_name for log_name in self.logs_desired if log_name in self.log_columns]
        for log in self.logs_recorded:
            self.fitting_data[log] = []

        # Add all params
        for param_set in self.params.values():
            for param_name in param_set.keys():
                self.fitting_data[param_name] = []

        # Add AUC param
        for models in self.models_defined:
            self.fitting_data[f"{models['prefix']}_AUC"] = []


    def initialize_model(self):

        # Stablish start index and end index in case they are not specified in the config file
        if self.index_start is None:
            self.index_start = 0
        if self.index_end is None:
            self.index_end = self.df_log["imgIndex"].iloc[-1]

        # Define groups of models in case the peaks are close
        self.group2model = self.create_groups()

        self.models = {}
        self.params = {}
        self.intervals = {}
        self.midpoints = {}

        initial_integration = self.read_initial_integration(self.path_scans)

        # For each group, create the composite_model, the composite_params and the 2theta range
        for group, model_list in self.group2model.items():
            composite_model = None
            composite_params = None

            lowest_2theta = 10**10
            highest_2theta = 0

            # First, find interval with lowest_2theta and highest_2theta and calculate midpoint
            for model_id in model_list:
                model_config = self.models_defined[model_id]

                # Define 2theta range
                lowest_2theta = min(lowest_2theta, model_config["2thlimits"]["min"])
                highest_2theta = max(highest_2theta, model_config["2thlimits"]["max"])

            # Give more margin to the data interval. Aprox = (2 * self.data_interval)
            lowest_2theta = lowest_2theta - self.data_interval
            highest_2theta = highest_2theta + self.data_interval

            lowest_index = self.findClosest(initial_integration[self.twoTh_col], lowest_2theta)
            highest_index = self.findClosest(initial_integration[self.twoTh_col], highest_2theta)
            index_interval = (lowest_index, highest_index)

            lowest_value = initial_integration[self.twoTh_col][lowest_index]
            highest_value = initial_integration[self.twoTh_col][highest_index]
            midpoint = (lowest_value + highest_value)/2

            for model_id in model_list:
                model_config = self.models_defined[model_id]
                model = getattr(models, model_config['model_type'])(prefix=model_config['prefix'] + '_')

                # Add some initial guesses, in this order.
                model = self.set_initial_hints(model, model_config, midpoint)
                params = self.set_initial_params(model, model_config, initial_integration, midpoint)

                if composite_model is None:
                    composite_model = model
                else:
                    composite_model += model

                if composite_params is None:
                    composite_params = params
                else:
                    composite_params.update(params)

                # We add the midpoint for every prefix. 
                # This is redundant in case two different models are in the same group, 
                # but it is handy to know the midpoint of each model.
                self.midpoints[model_config['prefix']] = midpoint
            
            
            # Give more margin to the data interval. Aprox = (2 * self.data_interval) + 0.1
            lowest_2theta = lowest_2theta - self.data_interval
            highest_2theta = highest_2theta + self.data_interval

            lowest_index = self.findClosest(initial_integration[self.twoTh_col], lowest_2theta)
            highest_index = self.findClosest(initial_integration[self.twoTh_col], highest_2theta)
            index_interval = (lowest_index, highest_index)


            # Give more margin to the data interval. Aprox = (2 * self.data_interval) + 0.1
            lowest_2theta = lowest_2theta - self.data_interval
            highest_2theta = highest_2theta + self.data_interval

            lowest_index = self.findClosest(initial_integration[self.twoTh_col], lowest_2theta)
            highest_index = self.findClosest(initial_integration[self.twoTh_col], highest_2theta)
            index_interval = (lowest_index, highest_index)

            # Add everything into the vectors
            self.models[group] = composite_model
            self.params[group] = composite_params
            self.intervals[group] = index_interval
            self.midpoints[group] = midpoint
    

    def create_groups(self):
        groups = [-1 for _ in range(len(self.models_defined))] # Value in position 'i' is the group of model 'i'
        next_group = 0

        # First we have to decide how are we going to group the models
        for i in range(self.num_models):
            if groups[i] == -1:
                groups[i] = next_group
                next_group += 1
            
            current_group_id = groups[i]

            for j in range(i+1, self.num_models):
                min_dist = min(
                    abs(self.models_defined[i]['2thlimits']['min'] - self.models_defined[j]['2thlimits']['min']),
                    abs(self.models_defined[i]['2thlimits']['min'] - self.models_defined[j]['2thlimits']['max']),
                    abs(self.models_defined[i]['2thlimits']['max'] - self.models_defined[j]['2thlimits']['min']),
                    abs(self.models_defined[i]['2thlimits']['max'] - self.models_defined[j]['2thlimits']['max'])
                )

                if min_dist < self.min_dist_for_group:
                    groups[j] = current_group_id

        group2model = {}

        for model, group in enumerate(groups):
            if group in group2model.keys():
                group2model[group].append(model)
            else:
                group2model[group] = [model]

        return group2model


    def find_integration_file(self, path_scans, target_index):
        list_files_integrations = os.listdir(path_scans)

        target_file = None

        for file in list_files_integrations:
            index = self.get_index_integration_file(file)
            if index == target_index:
                target_file = file
                break

        return target_file
    

    def read_initial_integration(self, path_scans):

        initial_integration_file = self.find_integration_file(path_scans, self.index_start)

        df_initial_integration = self.read_integrations_file(os.path.join(self.path_scans, initial_integration_file), self.facility)

        return df_initial_integration

    def read_integrations_file(self, filepath, facility = "ALBA"):
        if facility == "ALBA":
            return self.read_integrations_file_alba(filepath)
        else:
            return self.read_integrations_file_soleil(filepath)
            

    @staticmethod
    def read_integrations_file_alba(filepath):
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

    @staticmethod
    def read_integrations_file_soleil(filepath):
        df = pd.read_csv(filepath, sep = ' ')

        return df

    def get_index_integration_file(self, file):
        # All facilities share same system
        return int(file.split(".")[0].split("_")[-1])

    def set_initial_params(self, model, model_config, initial_integration, midpoint):
        default_params = {
            'center': (model_config['2thlimits']['min'] + model_config['2thlimits']['max'] - midpoint*2)*self.x_spacing/2 + self.x_spacing*0.005,
        }
        # default_params = {}
        params = model.make_params(**default_params)

        return params

    def set_initial_hints(self, model, model_config, midpoint):
        # model.set_param_hint('height', min=1e-10)
        model.set_param_hint('amplitude', min=1e-10)
        model.set_param_hint(
            'center',
            min=(model_config['2thlimits']['min'] - midpoint - self.data_interval) * self.x_spacing,
            max=(model_config['2thlimits']['max'] - midpoint + self.data_interval) * self.x_spacing,
            vary=True
        )

        return model

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
    parser = argparse.ArgumentParser(description='Fit peaks of an experiment according to some configuration.')
    parser.add_argument('yml_file', metavar='yml_file', type=str, nargs=1,
                    help='Path of the yaml file with the configuration of the desired peaks to fit')

    args = parser.parse_args()
    yml_file = args.yml_file[0]

    config_dict = get_yml_content(yml_file)

    peak_fitter = Peak_fitter(config_dict)

    peak_fitter.run_fitting()


if __name__ == "__main__":
    main()
