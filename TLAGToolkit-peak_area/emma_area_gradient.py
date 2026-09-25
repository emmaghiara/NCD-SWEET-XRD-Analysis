from tlagtk import peak_area_gradient as pf
import matplotlib.pyplot as plt
import os

yml_file = r'C:\Users\eghiara\Desktop\ALBA\NCD_Mar24\DATA\PROCESSED\YBCO\4mbar\Y12095\heating.yml'

config_dict = pf.get_yml_content(yml_file)

peak_fitter = pf.Peak_area_gradient(config_dict)

peak_fitter.run_fitting()