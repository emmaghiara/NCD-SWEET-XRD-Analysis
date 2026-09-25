from tlagtk import peak_area as pf
import matplotlib.pyplot as plt
import os

yml_file = r'C:\Users\eghiara\Desktop\ALBA\NCD_Feb23\DATA\PROCESSED\Y11292\final.yml'

config_dict = pf.get_yml_content(yml_file)

peak_fitter = pf.Peak_area(config_dict)

peak_fitter.run_fitting()