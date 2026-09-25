import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import matplotlib.cm as cm

# Function to select files using a file dialog
def select_files():
    root = Tk()
    root.withdraw()  # Hide the main tkinter window
    file_paths = filedialog.askopenfilenames(title="Select .dat Files", filetypes=[("DAT files", "*.dat")])
    return file_paths

# Prompt user to select .dat files
file_names = select_files()

if not file_names:
    print("No files selected. Exiting.")
    exit()

# Define a colormap that transitions from red to blue
# colors = ["red", "deeppink", "fuchsia", "darkviolet", "blue"]
colors = cm.get_cmap('plasma', len(file_names))

# Initialize a figure for the plot
plt.figure(figsize=(10, 6))

# Loop through each file, read the data, and plot it
for i, file_name in enumerate(file_names):
    try:
        # Load data from the file (assuming space-separated values)
        data = np.loadtxt(file_name)
        two_theta = data[:, 0]  # First column: 2theta
        intensity = data[:, 1]  # Second column: Intensity

        # Plot the data with a label
        plt.plot(two_theta, intensity, label=f"Dataset {i + 1}", color=colors(i), alpha=0.5)
        # plt.plot(two_theta, intensity, label=f"Point {i + 1}", color=colors[i % len(colors)], alpha=0.5)
    except Exception as e:
        print(f"Error reading file {file_name}: {e}")

# Customize the plot
plt.title("XRD Data Plot")
plt.xlabel("2θ (Degrees)")
plt.ylabel("Intensity (a.u.)")
plt.legend()
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
