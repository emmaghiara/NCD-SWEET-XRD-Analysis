import os
from tkinter import Tk, filedialog
import re

# Function to select a folder using a file dialog
def select_folder():
    root = Tk()
    root.withdraw()  # Hide the main tkinter window
    folder_path = filedialog.askdirectory(title="Select Folder Containing .dat Files")
    return folder_path

# Prompt user to select the folder containing .dat files
input_folder = select_folder()

if not input_folder:
    print("No folder selected. Exiting.")
    exit()

# Get all .dat files in the folder
dat_files = [f for f in os.listdir(input_folder) if f.endswith(".dat")]

if not dat_files:
    print("No .dat files found in the selected folder. Exiting.")
    exit()

# Sort the files to ensure consistent renaming
dat_files.sort()

# Perform the renaming
# base_number = 6671  # The number to rescale from
for i, file_name in enumerate(dat_files):
    old_path = os.path.join(input_folder, file_name)
    # file_parts = file_name.split("_")
    # Split by both '_' and '.'
    file_parts = re.split(r"[_\.]", file_name)

    # Replace the last part with the rescaled number, keeping leading zeros
    new_number = f"{i:04d}"
    # new_number = f"{0:03d}"
    # file_parts[-3] = f"{int(file_parts[-2]):03d}"
    file_parts[-2] = new_number

    # Reconstruct the new file name
    new_file_name = "_".join(file_parts[:-1]) + "." + file_parts[-1]
    new_path = os.path.join(input_folder, new_file_name)

    # Rename the file
    os.rename(old_path, new_path)
    print(f"Renamed: {file_name} -> {new_file_name}")

print("Renaming completed.")