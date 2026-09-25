from random import sample
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import pchip
import fabio
import os
import sys
import timeit

from collections import Counter


def get_instruments_logs(path_data, path_logs):
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
            return log_name
        

def read_logs(log_name, path_data, path_logs):
    df_log_temperature = None
    df_log_pressure = None
    df_log_resistance = None
    for log in log_name:
        # Read log temperature
        filename_temperature = "temperature_" + log + ".csv"
        df_log_temperature_i = pd.read_csv(path_data + path_logs + "temperature/" + filename_temperature, sep=',')
        if df_log_temperature is None:
            df_log_temperature = df_log_temperature_i
        else:
            df_log_temperature = pd.concat([df_log_temperature, df_log_temperature_i])

        # Read log pressure
        filename_pressure = "pressure_" + log + ".csv"
        df_log_pressure_i = pd.read_csv(path_data + path_logs + "pressure/" + filename_pressure, sep=',')
        if df_log_pressure is None:
            df_log_pressure = df_log_pressure_i
        else:
            df_log_pressure = pd.concat([df_log_pressure, df_log_pressure_i])

        # # Read log resistance if exists
        # file_resistance_path = path_data + path_logs + "resistance/resistance_" + log + ".csv"
        # if os.path.exists(file_resistance_path):
        #     df_log_resistance_i = pd.read_csv(file_resistance_path, sep=',')
        #     if df_log_resistance is None:
        #         df_log_resistance = df_log_resistance_i
        #     else:
        #         df_log_resistance = pd.concat([df_log_resistance, df_log_resistance_i])
        
    t0 = df_log_temperature["Time"].values[0] 
    df_log_temperature["Time_relative"] = df_log_temperature["Time"] - t0
    df_log_pressure["Time_relative"] = df_log_pressure["Time"] - t0
    df_log_pressure["Pressure_merged"] = [row["Pressure1"] if row["Pressure1"] < 1.3 else row["Pressure2"] for _, row in df_log_pressure.iterrows()]
    # if df_log_resistance is not None:
    #     df_log_resistance["Time_relative"] = df_log_resistance["Time"] - t0
        
    return df_log_temperature, df_log_pressure, df_log_resistance
    


def create_log(sample_name, path_data, path_logs, path_raw, path_save, instrument_logs):
    
    scans_id_set = get_scans(sample_name, path_data, path_raw)
    
    # concatenate temperature, pressure, resistance information from the logs
    logs_name = []
    for scan_id in scans_id_set:
        scan_images = get_scan_images(sample_name, scan_id, path_data, path_raw)
        first_image_name = scan_images[0]
        first_image_path = path_data + path_raw + sample_name + "/" + first_image_name
        first_image = fabio.open(first_image_path)
        t0_image_shift = float(first_image.header["time_of_day"])-float(first_image.header["time_of_frame"])
        
        log_name = get_corresponding_log_name(instrument_logs, t0_image_shift)
        if log_name is not None and log_name not in logs_name:
            logs_name.append(log_name)
        print("Using logs: ", logs_name)

    df_log_temperature, df_log_pressure, df_log_resistance = read_logs(logs_name, path_data, path_logs)
    
    # for each image, we get the index and timestamp, substract t0 and calculate interpolated temperature and pressure
    print(sample_name, "has the following scans: ", scans_id_set)
    for scan_id in scans_id_set:
        try:
            scan_images = get_scan_images(sample_name, scan_id, path_data, path_raw)
            first_image_name = scan_images[0]
            first_image_path = path_data + path_raw + sample_name + "/" + first_image_name
            first_image = fabio.open(first_image_path)
            t0_image_shift = float(first_image.header["time_of_day"])-float(first_image.header["time_of_frame"])

            # fit temperature, pressure, resistance
            temperature_fit = pchip(df_log_temperature["Time_relative"], df_log_temperature["Temperature"])
            pressure_fit = pchip(df_log_pressure["Time_relative"], df_log_pressure["Pressure_merged"])
            if df_log_resistance is not None:
                resistance_fit = pchip(df_log_resistance["Time_relative"], df_log_resistance["Measurement"])

            t0 = df_log_temperature["Time"].values[0]

            new_list_log = [None for _ in range(len(scan_images))]
            for i, image_name in enumerate(scan_images):
                
                image_log = {}

                image_index = image_name.split(".")[0].split("_")[-1]
                image_path = path_data + path_raw + sample_name + "/" + image_name
                image = fabio.open(image_path)

                timestamp = round(float(image.header["time_of_frame"]) + t0_image_shift, 8)
                image_time = round(timestamp - t0, 8)
                temperature = round(temperature_fit(image_time).item(), 5)
                pressure = round(pressure_fit(image_time).item(), 5)           
                
                # if df_log_resistance is not None:
                #     resistance = round(resistance_fit(image_time).item(), 5)
                image_att1 = image.header["att1"]
                image_att2 = image.header["att2"]
                image_omega = image.header["spitch"]

                if df_log_resistance is not None:
                    image_log = {"imgIndex" : image_index,
                                "scan" : scan_id,
                                "time" : image_time,
                                "timestamp" : timestamp,
                                "temperature" : temperature,
                                "pressure" : pressure,
                                # "resistance" : resistance,
                                "att1" : image_att1,
                                "att2" : image_att2,
                                "omega": image_omega
                                }
                else:
                    image_log = {"imgIndex" : image_index,
                                "scan" : scan_id,
                                "time" : image_time,
                                "timestamp" : timestamp,
                                "temperature" : temperature,
                                "pressure" : pressure,
                                "att1" : image_att1,
                                "att2" : image_att2,
                                "omega": image_omega
                                }

                new_list_log[i] = image_log

                
            #new_log_save_path = path_data + path_save + f"log_{sample_name}_{scan_id}.log"
            new_log_save_path = path_save + f"log_{sample_name}_{scan_id}.log"

            new_df_log = pd.DataFrame(new_list_log)
            new_df_log.to_csv(new_log_save_path, index=False, header=True, sep = ",")
            print("     Success merging logs of scan ", scan_id)
            print('     There is a shift of time between time_of_frame and time_of_day of final scan of:', round(float(image.header["time_of_day"])-timestamp,5), ' seconds')

        except Exception as e:
            print("     Failed to merge logs of scan ", scan_id)



def create_log_gradient(sample_name, exceptions_logs, path_data, path_logs, path_raw, path_save, gradients_start):
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

    num_scan_str = str(initial_scan).zfill(4)
    folder_name = sample_name + "_snapascan_" + num_scan_str + "/data_00/"
    path_scan = path_data + path_raw + sample_name + "/" + folder_name
    first_image_name = "rayonix_" + sample_name + num_scan_str + "_0000"
    first_image_path = path_data + path_raw + sample_name + "/" + folder_name + "/" + first_image_name
    first_image = fabio.open(first_image_path)
    t0_image_shift = float(first_image.header["time_of_day"])-float(first_image.header["time_of_frame"])
    
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

                #image_time = round(float(image.header["time_of_day"]) - t0, 5)
                timestamp = round(float(image.header["time_of_frame"]) + t0_image_shift, 6)
                image_time = round(timestamp - t0, 6)
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


def get_insitu_scan(sample_name, path_data, path_raw):
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


def get_scans(sample_name, path_data, path_raw):
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

    return sorted(scans_id)


def get_scan_images(sample_name, insitu_scan, path_data, path_raw):
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



def main():
    # sample = str(input("Enter the name of the sample: "))
    # drive_letter = str(input("Enter the drive letter where WD Discovery is located on your computer: "))
    # print("-------------------------------")
    # path_data = drive_letter + ":/WD Discovery/ALBA March 2024/DATA"
    # path_raw = "/RAW/"
    # path_logs = "/DEVICES_RAW/"
    # path_save = drive_letter + ":/WD Discovery/ALBA March 2024/DATA/PROCESSED/logs/"

    sample = "Y12038"
    path_data = "C:/Users/eghiara/Desktop/ALBA/NCD_Mar24/DATA/"
    path_raw = "/RAW/"
    path_logs = "/DEVICES_RAW/"
    path_save = "C:/Users/eghiara/Desktop/ALBA/NCD_Mar24/DATA/logs/"

    if not os.path.exists(path_data + path_raw):
        raise ValueError('The path for reading the RAW values does not exist')
    if not os.path.exists(path_data + path_logs):
        raise ValueError('The path for reading the logs does not exist')
    if not os.path.exists(path_save):
        raise ValueError('The path for saving the results does not exist')

    gradients_start = {
        "CIJP124" : ("0797", "1303"),
        "CIJP125" : ("2491", "3041"),
        "CIJP126" : ("0014", "0776"),
        "CIJP113" : ("1351", "1870"),
        "CIP117" : ("1351", "1870"),
        "CIJP119" : ("1935", "2422"),
    }
    
    arguments = sys.argv[1:]
    if len(arguments) == 1:
        target_sample = arguments[0]
    else:
        target_sample = "Y"

    instrument_logs = get_instruments_logs(path_data, path_logs)  # it have [t_start, t_end, log_num] for all logs with values

    print("-------------------------------")
    print("Trying with sample: ", sample)
    if sample.startswith(target_sample):
        try:
            print("Merging logs of sample", sample)
            create_log(sample, path_data, path_logs, path_raw, path_save, instrument_logs)
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
                create_log_gradient(sample, exceptions_logs, path_data, path_logs, path_raw, path_save gradients_start)
            except Exception as e:
                print("-------------------------------")
                print(sample)
                print(e)
        """


if __name__ == "__main__":
    main()

