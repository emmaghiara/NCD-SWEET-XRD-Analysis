import os
import shutil

def organize_files_by_suffix(input_root_folder, output_folder):
    """
    Traverse all subfolders of the input root folder and organize files into 
    folders based on the last three digits of their names.
    
    Args:
        input_root_folder (str): Path to the root folder containing the files.
        output_folder (str): Path to the folder where new folders will be created.
    """
    # Folder names based on suffix
    folder_suffixes = ['000', '001', '002', '003', '004']
    folder_paths = [os.path.join(output_folder, suffix) for suffix in folder_suffixes]

    # Create folders if they don't exist
    for folder_path in folder_paths:
        os.makedirs(folder_path, exist_ok=True)

    # Traverse the directory tree
    for dirpath, _, filenames in os.walk(input_root_folder):
        for file_name in filenames:
            # Construct full file path
            file_path = os.path.join(dirpath, file_name)

            # Skip directories; only process files
            if not os.path.isfile(file_path):
                continue

            # Extract the last three digits from the file name
            file_base_name = os.path.splitext(file_name)[0]  # Remove file extension
            suffix = file_base_name[-3:]  # Get last 3 characters

            # Determine target folder
            if suffix in folder_suffixes:
                target_folder = os.path.join(output_folder, suffix)
                shutil.copy(file_path, os.path.join(target_folder, file_name))
                print(f"Copied {file_name} to {target_folder}")
            else:
                print(f"Skipping {file_name}: Suffix {suffix} does not match any folder")

if __name__ == "__main__":
    # Define input root path and output path
    input_root_folder = r"C:\Users\eghiara\Desktop\ALBA\Beamtime Mar24\DATA\PROCESSED\Y12191a1\Y12191a1"
    output_folder = r"C:\Users\eghiara\Desktop\ALBA\Beamtime Mar24\DATA\PROCESSED\Y12191a1"

    # Run the organization script
    organize_files_by_suffix(input_root_folder, output_folder)
