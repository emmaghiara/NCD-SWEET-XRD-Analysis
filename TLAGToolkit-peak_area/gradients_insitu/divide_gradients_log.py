import os
from tkinter import Tk, filedialog

sample = 12191

# Function to select a file using a file dialog
def select_file():
    root = Tk()
    root.withdraw()  # Hide the main tkinter window
    file_path = filedialog.askopenfilename(title="Select a .log File", filetypes=[("Log files", "*.log")])
    return file_path

# Prompt user to select the input file
input_file = select_file()

if not input_file:
    print("No file selected. Exiting.")
    exit()

# Get the directory of the input file
input_dir = os.path.dirname(input_file)

# Read the input file and process its rows
output_files = {i: open(os.path.join(input_dir, f"{sample}log_{i}.log"), "w") for i in range(5)}  # Create 5 output files

try:
    with open(input_file, "r") as file:
        for line in file:
            try:
                # Parse the line and get the second column as an integer
                columns = line.split(',')
                if len(columns) > 1:
                    category = int(columns[1])
                    if 0 <= category < 5:
                        output_files[category].write(line)
            except ValueError:
                print(f"Skipping line due to parsing error: {line.strip()}")
finally:
    # Close all output files
    for f in output_files.values():
        f.close()

print("Files have been successfully split and saved.")