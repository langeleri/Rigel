import errors
import nights
import sources
import misc_functions

import csv
import glob
import numpy as np
import os
import pandas as pd
import time


t0 = time.perf_counter()
input_path = input("Welcome, Please input the target filepath. This filepath should point to a main directory containing sub directories for each observation night. Ex: ~/Desktop/Data_Directory/ ")
target_path = sorted(glob.glob((input_path + "*")))

print(target_path)
if len(target_path) == 0:
    raise Warning("No files found in target directory, please ensure correct formatting.")
 

print('Initializing data, this may take a moment depending on data volume and hardware.')
Nights = [nights.Night(directory, dir_number) for dir_number, directory in enumerate(target_path)]
for night in Nights:
    night.initialize_frames()
print("Data Initialization Complete")



print("Initializing Sources")
Sources = [sources.Source(source, count, Nights[0].wcs[0]) for count, source in enumerate(Nights[0].references)]
for source in Sources:
    source.query_source(Nights[0].obs_filter)
print("Source Initializiation Complete")


night_array = []
for night in Nights:
    for image in night.aligned_images:
        for source in Sources:
            source.boundary_check(night)
            if not source.flagged:
                source.aperture_photometry(image, night)
        night_array.append(night.obs_night)


threshold = input("Please Enter Desired Calibration Magnitude Threshold")
if threshold == "":
    threshold = 15 #set default value if step is skipped
else:
    threshold = float(threshold)



print("Calibrating Photometry")
counter = 0
slopes = []
zeros = []
MAEs = []
refined_source_list = [s for s in Sources if s.is_reference == True and s.ref_mag != None and s.ref_mag < threshold and s.flagged !=True]
for night in Nights:
    for image in night.aligned_images:
        means, sigs = errors.calculate_errors(refined_source_list, counter=counter, mag_thresh=threshold)
        slopes.append(means[0])
        zeros.append(means[1])
        mae_num = np.sum([abs(s.ref_mag - (means[0]*s.inst_mags[counter] + means[1])) for s in refined_source_list])
        mae_denom = np.sum([abs(s.ref_mag) for s in refined_source_list])
        MAEs.append(mae_num/mae_denom)
        counter += 1


for source in Sources:
    if not source.flagged:
        source.clear_calibration()
        for i in range(0, len(slopes)):
            mag = (source.inst_mags[i]*slopes[i] + zeros[i])
            final_err = source.inst_mag_errs[i]
            source.add_calibrated_mag(mag)
            source.add_error(final_err)


misc_functions.output_data(Nights, Sources, MAEs)



t1 = time.perf_counter()
print(f"Elapsed Time: {t1-t0} seconds. Images processed: {len(Nights)*25}")