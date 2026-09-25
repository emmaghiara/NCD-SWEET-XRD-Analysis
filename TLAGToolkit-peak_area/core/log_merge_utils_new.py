from random import sample
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import pchip
import fabio
import os
import sys

from collections import Counter

path_data = "/beamlines/bl11/projects/cycle2024-I/2023027451-epach/DATA"
path_raw = "/RAW/"
path_logs = "/DEVICES_RAW/"
path_save = "/PROCESSED/logs/"


# For experiment in RAW/
    # Create a log for each scan (000, 001, 002, 003) ...

exceptions_logs = {
    "Y10774a2" : "559",
    "Y10833" : "576",
}

gradients_start = {
    "CIJP124" : ("0797", "1303"),
    "CIJP125" : ("2491", "3041"),
    "CIJP126" : ("0014", "0776"),
    "CIJP113" : ("1351", "1870"),
    "CIP117" : ("1351", "1870"),
    "CIJP119" : ("1935", "2422"),
}

def get_instruments_logs():
    """
    Returns a list of three entries: start_time, finish_time, log_name
    """
    # Choose temperature as reference
    log_path = path_data+path_logs+"temperature"

    log_files = os.listdir(log_path)

    instrument_logs = []

    for filename_temperature in log_files:
        try:
            log_name = filename_temperature.split(".")[0].split("_")[-1]

            df_log_temperature = pd.read_csv(path_data + path_logs + "temperature/" + filename_temperature, sep=',')

            t0 = df_log_temperature["Time"].values[0]
            tf = df_log_temperature["Time"].values[-1]

            instrument_logs.append([t0, tf, log_name])
        except:
            print("Could not read log file", filename_temperature)

    return instrument_logs

def get_corresponding_log_name(instrument_logs, t0_image):
    for t0, tf, log_name in instrument_logs:
        if t0 <= t0_image and t0_image <= tf:
            print("log_found!")
            return log_name
        

def read_logs(log_name):
    # Read log temperature
    filename_temperature = "temperature_" + log_name + ".csv"
    df_log_temperature = pd.read_csv(path_data + path_logs + "temperature/" + filename_temperature, sep=',')

    # Read log pressure
    filename_pressure = "pressure_" + log_name + ".csv"
    df_log_pressure = pd.read_csv(path_data + path_logs + "pressure/" + filename_pressure, sep=',')

    # t0 = first register of log temperature
    t0 = df_log_temperature["Time"].values[0] 

    df_log_temperature["Time_relative"] = df_log_temperature["Time"] - t0
    df_log_pressure["Time_relative"] = df_log_pressure["Time"] - t0
    df_log_pressure["Pressure_merged"] = [row["Pressure1"] if row["Pressure1"] < 1.3 else row["Pressure2"] for _, row in df_log_pressure.iterrows()]

    return df_log_temperature, df_log_pressure

def create_log(sample_name):
    
    if sample_name in exceptions_logs.keys():
        log_name = exceptions_logs[sample_name] + ".csv"
    else:
        log_name = sample_name + ".csv"

    # plt.plot(df_log_pressure["Time_relative"], df_log_pressure["Pressure_merged"]*25)
    # plt.plot(df_log_temperature["Time_relative"], df_log_temperature["Temperature"])
    # plt.show()
    scans_id_set = get_scans(sample_name)
    print(sample_name, "has the following scans: ", scans_id_set)
    for scan_id in scans_id_set:
        # insitu_scan = "005"

        # for each image, 
        #    get the index and timestamp, substract t0 and calculate interpolated temperature and pressure
        #    put this info in a data frame

        # save df 
        try:
            scan_images = get_scan_images(sample_name, scan_id)
            first_image_name = scan_images[0]
            first_image_path = path_data + path_raw + sample_name + "/" + first_image_name
            first_image = fabio.open(first_image_path)
            t0_image = float(first_image.header["time_of_day"])
            
            log_name = get_corresponding_log_name(instrument_logs, t0_image)

            df_log_temperature, df_log_pressure = read_logs(log_name)

            # fit temperature
            temperature_fit = pchip(df_log_temperature["Time_relative"], df_log_temperature["Temperature"])

            # fit pressure
            pressure_fit = pchip(df_log_pressure["Time_relative"], df_log_pressure["Pressure_merged"])

            t0 = df_log_temperature["Time"].values[0]

            new_list_log = [None for _ in range(len(scan_images))]
            for i, image_name in enumerate(scan_images):
                
                image_log = {}

                image_index = image_name.split(".")[0].split("_")[-1]

                image_path = path_data + path_raw + sample_name + "/" + image_name
                image = fabio.open(image_path)

                timestamp = float(image.header["time_of_day"])
                image_time = round(timestamp - t0, 5)
                temperature = round(temperature_fit(image_time).item(), 5)
                pressure = round(pressure_fit(image_time).item(), 5)

                image_log = {"imgIndex" : image_index,
                            "scan" : scan_id,
                            "time" : image_time,
                            "timestamp" : timestamp,
                            "temperature" : temperature,
                            "pressure" : pressure
                            }

                new_list_log[i] = image_log

                
            new_log_save_path = path_data + path_save + f"log_{sample_name}_{scan_id}.log"

            new_df_log = pd.DataFrame(new_list_log)
            new_df_log.to_csv(new_log_save_path, index=False, header=True, sep = ",")
            print("     Success merging logs of scan ", scan_id)
        except Exception as e:
            print("     Failed to merge logs of scan ", scan_id)

def create_log_gradient(sample_name):
    if sample_name in exceptions_logs.keys():
        log_name = exceptions_logs[sample_name] + ".csv"
    else:
        log_name = sample_name + ".csv"

    # Read log temperature
    filename_temperature = "temperature_" + log_name
    df_log_temperature = pd.read_csv(path_data + path_logs + "temperature/" + filename_temperature, sep=',')

    # Read log pressure
    filename_pressure = "pressure_" + log_name
    df_log_pressure = pd.read_csv(path_data + path_logs + "pressure/" + filename_pressure, sep=',')

    # t0 = first register of log temperature
    t0 = df_log_temperature["Time"].values[0]

    df_log_temperature["Time_relative"] = df_log_temperature["Time"] - t0
    df_log_pressure["Time_relative"] = df_log_pressure["Time"] - t0
    df_log_pressure["Pressure_merged"] = [row["Pressure1"] if row["Pressure1"] < 1.3 else row["Pressure2"] for _, row in df_log_pressure.iterrows()]

    # fit temperature
    temperature_fit = pchip(df_log_temperature["Time_relative"], df_log_temperature["Temperature"])

    # fit pressure
    pressure_fit = pchip(df_log_pressure["Time_relative"], df_log_pressure["Pressure_merged"])

    initial_scan, final_scan = gradients_start[sample_name]
    initial_scan, final_scan = int(initial_scan), int(final_scan)
    
    new_list_log = []
    for i, num_scan in enumerate(range(initial_scan, final_scan + 1)):
        
        num_scan_str = str(num_scan).zfill(4)
        folder_name = sample_name + "_snapascan_" + num_scan_str + "/data_00/"

        path_scan = path_data + path_raw + sample_name + "/" + folder_name

        gradient_scans = sorted(os.listdir(path_scan))

        for gradient_scan in gradient_scans:

            if gradient_scan.split(".")[-1] == "edf":
            
                image_log = {}

                # image_index = image_name.split(".")[0].split("_")[-1]

                image_path = path_scan + gradient_scan
                image = fabio.open(image_path)

                image_time = round(float(image.header["time_of_day"]) - t0, 5)
                image_position = int(image.header["pt_no"])
                image_sx_position = round(float(image.header["sx"]), 5)

                if image_time < 0:
                    temperature = 0
                    pressure = 0
                else:
                    temperature = round(temperature_fit(image_time).item(), 5)
                    pressure = round(pressure_fit(image_time).item(), 5)

                image_log = {"imgIndex" : num_scan,
                            "position" : image_position,
                            "sx_position" : image_sx_position,
                            "scan" : initial_scan,
                            "sample" : sample_name,
                            "time" : image_time,
                            "temperature" : temperature,
                            "pressure" : pressure
                            }

                new_list_log.append(image_log)

    
    new_log_save_path = path_data + path_save + f"log_{sample_name}_{initial_scan}.log"
    
    new_df_log = pd.DataFrame(new_list_log)
    new_df_log.to_csv(new_log_save_path, index=False, header=True, sep = ",")

    num_positions = max(new_df_log["position"]) + 1
    for position in range(num_positions):
        df_log_individual_position = new_df_log[new_df_log["position"] == position]

        new_log_save_path = path_data + path_save + f"log_{sample_name}_{initial_scan}_p{position}.log"

        df_log_individual_position.to_csv(new_log_save_path, index=False, header=True, sep = ",")
        print(position, len(df_log_individual_position))


def get_insitu_scan(sample_name):
    """
    Get scan id with higher number of images. It can be one scan or two scans (in case of samples where the acquisition time has been changed during the experiment)
    """
    path_sample = path_data + path_raw + sample_name

    items = os.listdir(path_sample)

    image_counter = Counter()

    for item in items:
        if item.split(".")[-1] == "edf":
            scan_number = item.split("_")[-2]
            if scan_number in image_counter.elements():
                image_counter[scan_number] += 1
            else:
                image_counter[scan_number] = 1

    # for item in items:
    #     if (len(item)==3 and item.isnumeric()) or (len(item.split("&")) == 2):
    #         num_files = len(os.listdir(path_sample + "/" + item))
        
    #         image_counter[item] = num_files
            

    most_common_scan = image_counter.most_common(1)[0][0]
    
    # insitu_scans = []

    # for scan_id, num_images in most_common_scans:
    #     if num_images > 300:
    #         insitu_scans.append(scan_id)

    return most_common_scan

def get_scans(sample_name):
    """
    Get list of scans ids
    """
    path_sample = path_data + path_raw + sample_name

    items = os.listdir(path_sample)

    scans_id = set()

    for item in items:
        if item.split(".")[-1] == "edf":
            scan_number = item.split("_")[-2]
            scans_id.add(scan_number)

    return scans_id


def get_scan_images(sample_name, insitu_scan):

    """
    path_sample = path_data + path_raw + sample_name + "/" + insitu_scan

    items = os.listdir(path_sample)

    scan_images = []

    for item in items:
        if item.split(".")[-1] == "edf":
            scan_images.append(item)
    """
    path_images = path_data + path_raw + sample_name
    
    items = os.listdir(path_images)
    
    scan_images = []

    for item in items:
        if item.split(".")[-1] == "edf" and item.split("_")[-2] == insitu_scan:
            
            scan_images.append(item)
    
    scan_images.sort()
    
    return scan_images


arguments = sys.argv[1:]
print(arguments)
if len(arguments) == 1:
    target_sample = arguments[0]
else:
    target_sample = "Y"

sample_list = os.listdir(path_data + path_raw)
logs_list = os.listdir(path_data + path_save)
logs_list_sample = [log.split("_")[1] for log in logs_list]

instrument_logs = get_instruments_logs()

for sample in sample_list:
    print("-------------------------------")
    print("Trying with sample: ", sample)
    if sample.startswith(target_sample):
        try:
            print("Merging logs of sample", sample)
            create_log(sample)
        except Exception as e:
            print("-------------Error-------------")
            print(sample)
            print(e)
            raise e
            print("-------------------------------")
    """
    elif sample[0] == "C":
        try:
            print("Merging logs of sample", sample)
            create_log_gradient(sample)
        except Exception as e:
            print("-------------------------------")
            print(sample)
            print(e)
    """

"""
# hardcoded logging

sample_name = "Y10785"
create_log(sample_name)

"""
# BEAMTIME_DATAPATH = os.path.dirname(__file__)
# RAW = "RAW"
# DEVICES_RAW_PATH = os.path.join(BEAMTIME_DATAPATH, "DEVICES_RAW")
# LOG_MERGE_SCRIPT = os.path.join(BEAMTIME_DATAPATH, "code_analysis/log_merge_utils.py")
# TLAG_RESULTS_PATH = "/homelocal/opbl11/Documents/TLAG/tlag_manager/results/*"
# SAVING_PATH = os.path.join(BEAMTIME_DATAPATH, "PROCESSED", "logs")


# class Scripts():
#     def copy_tlag_data(self):
#         if not os.path.exists(LOG_MERGE_SCRIPT):
#             print("ERROR! File does not exists: %s"%LOG_MERGE_SCRIPT)
#             return
        
#         print("Copying data from ctbltlag01")
#         os.system('scp -r opbl11@ctbltlag01:%s %s' %(TLAG_RESULTS_PATH, DEVICES_RAW_PATH))
#         print("Running merge script")
#         os.system('exec /usr/local/bin/conda-exec TLAG-analysis python3 %s' %LOG_MERGE_SCRIPT)
        
        
#     def create_logs(self):
#         self.copy_tlag_data()
#         sample_list = os.listdir(os.path.join(BEAMTIME_DATAPATH, RAW))

#         for sample in sample_list:
#             print("Trying with sample: ", sample)
#             # if sample[0] == "Y":
#             try:
#                 print("Merging logs of sample", sample)
#                 self._create_log(sample)
#             except Exception as e:
#                 print("-------------Error-------------")
#                 print(sample)
#                 print(e)
#                 print("-------------------------------")
                
#     def _create_log(sample_name):
#         if sample_name in exceptions_logs.keys():
#             log_name = exceptions_logs[sample_name] + ".csv"
#         else:
#             log_name = sample_name + ".csv"

#         # Read log temperature
#         filename_temperature = "temperature_" + log_name
#         df_log_temperature = pd.read_csv(os.path.join(DEVICES_RAW_PATH, "temperature", filename_temperature), sep=',')

#         # Read log pressure
#         filename_pressure = "pressure_" + log_name
#         df_log_pressure = pd.read_csv(os.path.join(DEVICES_RAW_PATH, "pressure", filename_pressure), sep=',')

#         # t0 = first register of log temperature
#         t0 = df_log_temperature["Time"].values[0] 

#         df_log_temperature["Time_relative"] = df_log_temperature["Time"] - t0
#         df_log_pressure["Time_relative"] = df_log_pressure["Time"] - t0
#         df_log_pressure["Pressure_merged"] = [row["Pressure1"] if row["Pressure1"] < 1.3 else row["Pressure2"] for _, row in df_log_pressure.iterrows()]

#         # fit temperature
#         temperature_fit = pchip(df_log_temperature["Time_relative"], df_log_temperature["Temperature"])

#         # fit pressure
#         pressure_fit = pchip(df_log_pressure["Time_relative"], df_log_pressure["Pressure_merged"])

#         # plt.plot(df_log_pressure["Time_relative"], df_log_pressure["Pressure_merged"]*25)
#         # plt.plot(df_log_temperature["Time_relative"], df_log_temperature["Temperature"])
#         # plt.show()
#         insitu_scan = self._get_insitu_scan(sample_name)
#         # insitu_scan = "003"

#         # for each image, 
#         #    get the index and timestamp, substract t0 and calculate interpolated temperature and pressure
#         #    put this info in a data frame

#         # save df 
#         scan_images = self._get_scan_images(sample_name, insitu_scan)
#         new_list_log = [None for _ in range(len(scan_images))]
#         for i, image_name in enumerate(scan_images):
            
#             image_log = {}

#             image_index = image_name.split(".")[0].split("_")[-1]

#             image_path = os.path.join(BEAMTIME_DATAPATH, RAW, sample_name, image_name)
#             image = fabio.open(image_path)

#             timestamp = float(image.header["time_of_day"])
#             image_time = round(timestamp - t0, 5)
#             temperature = round(temperature_fit(image_time).item(), 5)
#             pressure = round(pressure_fit(image_time).item(), 5)

#             image_log = {"imgIndex" : image_index,
#                         "scan" : insitu_scan,
#                         "time" : image_time,
#                         "timestamp" : timestamp,
#                         "temperature" : temperature,
#                         "pressure" : pressure
#                         }

#             new_list_log[i] = image_log

            
#         new_log_save_path = os.path.join(SAVING_PATH,f"log_{sample_name}_{insitu_scan}.log")

#         new_df_log = pd.DataFrame(new_list_log)
#         new_df_log.to_csv(new_log_save_path, index=False, header=True, sep = ",")
        
#     def create_log_gradient(sample_name):
#         if sample_name in exceptions_logs.keys():
#             log_name = exceptions_logs[sample_name] + ".csv"
#         else:
#             log_name = sample_name + ".csv"

#          # Read log temperature
#         filename_temperature = "temperature_" + log_name
#         df_log_temperature = pd.read_csv(os.path.join(DEVICES_RAW_PATH, "temperature", filename_temperature), sep=',')

#         # Read log pressure
#         filename_pressure = "pressure_" + log_name
#         df_log_pressure = pd.read_csv(os.path.join(DEVICES_RAW_PATH, "pressure", filename_pressure), sep=',')


#         # t0 = first register of log temperature
#         t0 = df_log_temperature["Time"].values[0]

#         df_log_temperature["Time_relative"] = df_log_temperature["Time"] - t0
#         df_log_pressure["Time_relative"] = df_log_pressure["Time"] - t0
#         df_log_pressure["Pressure_merged"] = [row["Pressure1"] if row["Pressure1"] < 1.3 else row["Pressure2"] for _, row in df_log_pressure.iterrows()]

#         # fit temperature
#         temperature_fit = pchip(df_log_temperature["Time_relative"], df_log_temperature["Temperature"])

#         # fit pressure
#         pressure_fit = pchip(df_log_pressure["Time_relative"], df_log_pressure["Pressure_merged"])

#         initial_scan, final_scan = gradients_start[sample_name]
#         initial_scan, final_scan = int(initial_scan), int(final_scan)
        
#         new_list_log = []
#         for i, num_scan in enumerate(range(initial_scan, final_scan + 1)):
            
#             num_scan_str = str(num_scan).zfill(4)
#             folder_name = sample_name + "_snapascan_" + num_scan_str + "/data_00/"

#             path_scan = os.path.join(BEAMTIME_DATAPATH,  RAW, sample_name,folder_name)

#             gradient_scans = sorted(os.listdir(path_scan))

#             for gradient_scan in gradient_scans:

#                 if gradient_scan.split(".")[-1] == "edf":
                
#                     image_log = {}

#                     # image_index = image_name.split(".")[0].split("_")[-1]

#                     image_path = os.path.join(path_scan,gradient_scan)
#                     image = fabio.open(image_path)

#                     image_time = round(float(image.header["time_of_day"]) - t0, 5)
#                     image_position = int(image.header["pt_no"])
#                     image_sx_position = round(float(image.header["sx"]), 5)

#                     if image_time < 0:
#                         temperature = 0
#                         pressure = 0
#                     else:
#                         temperature = round(temperature_fit(image_time).item(), 5)
#                         pressure = round(pressure_fit(image_time).item(), 5)

#                     image_log = {"imgIndex" : num_scan,
#                                 "position" : image_position,
#                                 "sx_position" : image_sx_position,
#                                 "scan" : initial_scan,
#                                 "sample" : sample_name,
#                                 "time" : image_time,
#                                 "temperature" : temperature,
#                                 "pressure" : pressure
#                                 }

#                     new_list_log.append(image_log)

        
#         new_log_save_path = os.path.join(SAVING_PATH, f"log_{sample_name}_{initial_scan}.log")
        
#         new_df_log = pd.DataFrame(new_list_log)
#         new_df_log.to_csv(new_log_save_path, index=False, header=True, sep = ",")

#         num_positions = max(new_df_log["position"]) + 1
#         for position in range(num_positions):
#             df_log_individual_position = new_df_log[new_df_log["position"] == position]

#             new_log_save_path = os.path.join(SAVING_PATH, f"log_{sample_name}_{initial_scan}_p{position}.log")

#             df_log_individual_position.to_csv(new_log_save_path, index=False, header=True, sep = ",")
#             print(position, len(df_log_individual_position))
        
        
    
#     def _get_insitu_scan(sample_name):
#         """
#         Get scan id with higher number of images. It can be one scan or two scans (in case of samples where the acquisition time has been changed during the experiment)
#         """
#         path_sample = os.path.join(BEAMTIME_DATAPATH, RAW, sample_name)

#         items = os.listdir(path_sample)

#         image_counter = Counter()

#         for item in items:
#             if item.split(".")[-1] == "edf":
#                 scan_number = item.split("_")[-2]
#                 if scan_number in image_counter.elements():
#                     image_counter[scan_number] += 1
#                 else:
#                     image_counter[scan_number] = 1

#         # for item in items:
#         #     if (len(item)==3 and item.isnumeric()) or (len(item.split("&")) == 2):
#         #         num_files = len(os.listdir(path_sample + "/" + item))
            
#         #         image_counter[item] = num_files
                

#         most_common_scan = image_counter.most_common(1)[0][0]
        
#         # insitu_scans = []

#         # for scan_id, num_images in most_common_scans:
#         #     if num_images > 300:
#         #         insitu_scans.append(scan_id)

#         return most_common_scan
    

#     def _get_scan_images(sample_name, insitu_scan):

#         """
#         path_sample = path_data + path_raw + sample_name + "/" + insitu_scan

#         items = os.listdir(path_sample)

#         scan_images = []

#         for item in items:
#             if item.split(".")[-1] == "edf":
#                 scan_images.append(item)
#         """
#         path_images = os.path.join(BEAMTIME_DATAPATH, RAW, sample_name)
        
#         items = os.listdir(path_images)
        
#         scan_images = []
        
#         for item in items:
#             if item.split(".")[-1] == "edf" and item.split("_")[-2] == insitu_scan:
#                 scan_images.append(item)
        
#         scan_images.sort()
        
#         return scan_images
