from random import sample
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import pchip
import fabio
import os
import sys

def replace_in_log_file(file_path):
    # Read the contents of the file
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Replace '7763' with '001'
    modified_content = content.replace('7686', '001')
    
    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.write(modified_content)

# Example usage
log_file_path = 'C:/Users/eghiara/Desktop/ALBA/Beamtime Mar24/DATA/PROCESSED/Y12191/logs/log_Y12191a1_7686.log'
replace_in_log_file(log_file_path)