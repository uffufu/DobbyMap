''' 
Required Packages
- numpy: 1.19.5  
- matplotlib: 3.5.3  
- astropy: 5.1  
- spectral_cube: 0.5.0
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ===== cube reading and spectrum extraction =====
from spectral_cube import SpectralCube
from scipy.ndimage import gaussian_filter1d
from astropy.io import fits
from astropy import units as u

# ===== file handling =====
import os
from glob import glob
import tkinter as tk
from tkinter import filedialog
from datetime import datetime

'''
File Selection Functions: select a single FITS file or all FITS files in a folder

Functions
- select_file(): Opens a dialog to select a single FITS file. Returns the `file_path` as a string.
- select_folder_and_find_fits(): Opens a dialog to select a folder, then finds all FITS files in that folder. 

Return
- `file_path`: str
- `file_list`: list, processed in a loop in the main program.
- `folder_name`:str or None (without path), can be used for Excel worksheet name
'''

def select_file():
    # ===== user selects a single FITS file. =====
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select FITS file", filetypes=[("FITS files", "*.fits")])
    return file_path

def select_folder_and_find_fits():
    # ===== user selects a folder, and the script finds all FITS files in that folder. =====
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select folder")
    if not folder_path:
        print("No folder selected.")
        return [], None

    fits_files = glob(os.path.join(folder_path, "*.fits"))
    if not fits_files:
        print("No FITS files found in the selected folder.")
        return [], None

    folder_name = os.path.basename(folder_path)
    print("FITS files found：")
    for i, f in enumerate(fits_files):
        print(f"{i}: {f}")

    return fits_files, folder_name

'''
Read The Cube and Collect Central Spectrum 

Parameters
- `file_name` (str): Path to the FITS cube file. 
- `window` (int, optional): Length of the center region, the average is taken from the center `window x window` region.
If no value is input for `window`, the default value of 5 will be used.

Return values
- `spectrum` (numpy.ndarray): One-dimensional spectrum after averaging the center region (one value per channel).
- `data` (numpy.ndarray): The original cube data array (channel, y, x).
- `n_chan` (int): Number of channels.
- `ny` (int): Number of pixels along the y-axis.
- `nx` (int): Number of pixels along the x-axis.
- `cube` (SpectralCube): The SpectralCube object itself.
- `base` (str): The base filename (without extension), extracted from `file_name`.
- `restfreq` (float): Rest frequency read from the FITS header (`RESTFRQ` keyword). If not found, returns `np.nan`.
'''

def get_center_spectrum(file_name, window = 5):
    # ===== read cube and collect center spectrum =====
    cube = SpectralCube.read(file_name)
    data = cube.unmasked_data[:].value
    n_chan, ny, nx = data.shape
    base = os.path.splitext(os.path.basename(file_name))[0]
    header = fits.getheader(file_name)
    restfreq = header.get('RESTFRQ', np.nan)

    print(f"The total number of channels in this {file_name} is {n_chan}, width is {ny} pixels and height is {nx} pixels.")
    
    # ===== average of the central 5x5 area =====
    cy, cx = ny // 2, nx // 2
    subcube = data[:, cy - window//2:cy + window//2, cx - window//2:cx + window//2]
    spectrum = np.nanmean(subcube, axis=(1, 2))
    return spectrum, data, n_chan, ny, nx, cube, base, restfreq


'''
Smoothness and Slope: calculate the first derivative to track changes in slope  

Parameters
- `spectrum` (numpy.ndarray): One-dimensional spectrum after averaging the center region (one value per channel).
- `sigma` (float, optional): Standard deviation for Gaussian smoothing. Controls the degree of smoothing. Default is 1.5.
- `thresh_ratio` (float, optional): Ratio for the slope threshold. Default is 0.1.

Return values
- `spectrum_smooth` (numpy.ndarray): Spectrum after Gaussian smoothing.
- `slope` (numpy.ndarray): First derivative (slope) of the smoothed spectrum.
- `threshold_dy` (float): Slope threshold value, used for signal segment detection.
'''
def smooth_and_slope(spectrum, sigma=1.5, thresh_ratio=0.1):
    # ===== calculate the first derivative to track changes in slope =====

    # ===== smooth the spectrum with gaussian_filter1d =====
    spectrum_smooth = gaussian_filter1d(spectrum, sigma)

    # ===== calculate the slope =====
    slope = np.gradient(spectrum_smooth)

    # ===== set threshold for slope detection =====
    threshold_dy = np.max(np.abs(slope)) *  thresh_ratio
    return spectrum_smooth, slope, threshold_dy

'''
Find Emission Ranges  
Detects emission signal regions based on the slope of the spectrum.

Parameters
- `slope` (numpy.ndarray): The first derivative (slope) of the smoothed spectrum.
- `threshold_dy` (float): Slope threshold value for detecting signal regions.
- `min_length` (int, default=5): The minimum length of a detected emission region. Regions shorter than this value will be ignored.  

Return values
- `keep_ranges` (list of tuple): List of (start, end) index tuples, each representing a detected emission region in channel indices.
'''

def find_emission_ranges(slope, threshold_dy, min_length=5):
    # ===== find emission ranges based on slope =====
    keep_ranges = []
    in_signal, start = False, None

    for i in range(1, len(slope) - 1):
        if not in_signal and slope[i] > threshold_dy:
            in_signal, start = True, i
        elif in_signal and slope[i] < -threshold_dy:
            for j in range(i, len(slope)):
                if abs(slope[j]) < threshold_dy:
                    end = j
                    if end - start > min_length:
                        # ===== append the range if it is long enough =====
                        keep_ranges.append((start, end))
                    in_signal = False
                    break
    print("Channel detected：")
    for r in keep_ranges:
        print(f"Channel {r[0]} to {r[1]}")
    return keep_ranges


'''
Filter Range and Plot Spectrum  
Filters out emission ranges near the spectrum edges, plots the spectrum and slope with detected regions,
and returns filtered ranges, rest frequency, and channel range string.

Parameters
- `keep_ranges` (list of tuple): List of (start, end) index tuples representing detected emission regions.
- `n_chan` (int): Total number of channels in the spectrum.
- `spectrum` (numpy.ndarray): Original one-dimensional spectrum.
- `spectrum_smooth` (numpy.ndarray): Smoothed spectrum.
- `slope` (numpy.ndarray): First derivative (slope) of the smoothed spectrum.
- `threshold_dy` (float): Slope threshold value.
- `file_name` (str): Path to the FITS cube file (for reading header metadata).
- `fraction` (int, optional): Denominator for calculating edge width. Default is 8.
- `base` (str): The base filename (without extension), extracted from `file_name`.  

Return values
- `filtered_ranges` (list of tuple): List of (start, end) index tuples after filtering out edge regions.
- `restfreq` (float): Rest frequency read from FITS header (`RESTFRQ` keyword).
- `channel_ranges_str` (str): String representation of filtered channel ranges, e.g. `"12-34; 56-78"`.
'''

def filter_ranges(keep_ranges, n_chan, spectrum, spectrum_smooth, slope, threshold_dy, file_name, fraction =8, base =None):
    
    #Filter out ranges that could be the emission from other molecules,
    #plot spectrum and slope, and return filtered ranges, restfreq, and channel range string.
    
    edge = n_chan // fraction
    print(f"Exclude {edge} channels from each side of the spectrum")

    filtered_ranges = []
    for start, end in keep_ranges:
        if end < edge or start > n_chan - edge - 1:
            continue
        start_new = max(start, edge)
        end_new = min(end, n_chan - edge - 1)
        if end_new - start_new > 2:
            filtered_ranges.append((start_new, end_new))

    # plot the spectrum and slope
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(spectrum, label='Original')
    plt.plot(spectrum_smooth, label='Smoothed')
    for start, end in keep_ranges:
        plt.axvspan(start, end, color='yellow', alpha=0.3, label='Detected Peak')
    for i, (start, end) in enumerate(filtered_ranges):
        plt.axvline(start, color='green', linestyle='--', alpha=0.8, label='Filtered Range' if i == 0 else "")
        plt.axvline(end, color='green', linestyle='--', alpha=0.8)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.ylabel("Intensity")
    plt.title(f"{base} Spectrum with Detected Ranges")

    plt.subplot(2, 1, 2)
    plt.plot(slope, label='1st Derivative', color='hotpink')
    plt.axhline(threshold_dy, color='cornflowerblue', linestyle='--', label='+Threshold')
    plt.axhline(-threshold_dy, color='cornflowerblue', linestyle='--', label='-Threshold')
    plt.legend()
    plt.xlabel("Channel")
    plt.ylabel("dI/dv")
    plt.title(f"{base} 1st Derivative of Smoothed Spectrum")
    plt.tight_layout()
    plt.show()

    # read restfreq
    header = fits.getheader(file_name)
    restfreq = header.get('RESTFRQ', np.nan)

    # arrange channel range string
    channel_ranges_str = '; '.join([f"{start}-{end}" for start, end in filtered_ranges])

    return filtered_ranges, restfreq, channel_ranges_str

'''
Create RMS Noise Mask and Calculate RMS  
Creates a mask to exclude signal and edge channels, then calculates the RMS noise from the remaining data.

Parameters
- `data` (numpy.ndarray): The original cube data array (channel, y, x).
- `filtered_ranges` (list of tuple): List of (start, end) index tuples representing signal regions to exclude.
- `n_chan` (int): Total number of channels in the spectrum.
- `edge` (int): Number of channels to exclude at both the beginning and end of the spectrum.

Return values
- `noise_mask` (numpy.ndarray): Boolean mask array for channels used in RMS calculation.
- `rms` (float): Estimated RMS noise value from the masked data.
'''

def create_rms_mask(data, filtered_ranges, n_chan, edge):
    # ===== Create RMS noise mask and calculate RMS =====
    noise_mask = np.ones(n_chan, dtype=bool)
    noise_mask[:edge] = False
    noise_mask[-edge:] = False
    for start, end in filtered_ranges:
        noise_mask[start:end+1] = False

    rms = np.nanstd(data[noise_mask, :, :])
    print(f"Estimated RMS: {rms:.6g}")
    return noise_mask, rms

'''
Create Signal Mask  
Creates a mask for signal regions based on filtered ranges and RMS threshold.

Parameters
- `data` (numpy.ndarray): The original cube data array (channel, y, x).
- `filtered_ranges` (list of tuple): List of (start, end) index tuples representing signal regions.
- `rms` (float): Estimated RMS noise value.
- `number_sigma` (int or float, optional): Threshold multiplier for RMS. Data greater than `number_sigma * rms` is considered signal. Default is 3.

Return values
- `cube_mask` (numpy.ndarray): Boolean mask array where signal regions above the threshold are marked as True.
'''

def create_signal_mask(data, filtered_ranges, rms, number_sigma=3):
    """Create signal mask based on filtered ranges and RMS"""
    cube_mask = np.zeros_like(data, dtype=bool)
    for start, end in filtered_ranges:
        cube_mask[start:end+1, :, :] = data[start:end+1, :, :] > number_sigma * rms
    return cube_mask

'''
Calculate and Save Moment Maps  
Calculates moment 0, 1, and 2 from the masked cube and saves them with filenames based on the original FITS file.

Parameters
- `cube_masked` (SpectralCube): The masked spectral cube for moment calculation.
- `base` (str): The base filename (without extension), extracted from `file_name` (used for output filenames).

Return values
- `moment0` (SpectralCube): Moment 0 map (integrated intensity).
- `moment1` (SpectralCube): Moment 1 map (intensity-weighted velocity).
- `moment2_disp` (SpectralCube): Moment 2 map (velocity dispersion).

Output files
- Saves three FITS files:  
  - `<original_name>_moment0_<time>.fits`  
  - `<original_name>_moment1_<time>.fits`  
  - `<original_name>_moment2_<time>.fits`
'''

def calculate_moments(cube_masked, base):
    # Calculate moment 0, 1, and 2 from the masked cube
    moment0 = cube_masked.moment(order=0)
    moment1 = cube_masked.moment(order=1)
    moment2 = cube_masked.moment(order=2) 
    # Consistent with CARTA units
    if moment0.unit.is_equivalent(u.Jy / u.beam * u.m / u.s):
        moment0 = moment0.to(u.Jy / u.beam * u.km / u.s)
    if moment1.unit.is_equivalent(u.m / u.s):
        moment1 = moment1.to(u.km / u.s)
    if moment2.unit.is_equivalent((u.km / u.s)**2):
        moment2_disp = moment2 ** 0.5
        moment2_disp = moment2_disp.to(u.km / u.s)

    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # ===== save moment maps =====

    moment0.write(f"{base}_moment0_{timestamp}.fits", overwrite=True)
    moment1.write(f"{base}_moment1_{timestamp}.fits", overwrite=True)
    moment2_disp.write(f"{base}_moment2_{timestamp}.fits", overwrite=True)
    print("Moment 0/1/2 saved successfully")
    
    return moment0, moment1, moment2_disp

'''
Write Results to Excel

Parameters
- `file_name` (str): Name of the FITS file being processed.
- `restfreq` (float): Rest frequency of the spectrum in Hz.
- `rms` (float): RMS noise level of the spectrum.
- `filtered_ranges` (list of tuple): Filtered signal ranges as (start, end) tuples.
- `channel_ranges_str` (str): String representation of the filtered ranges.
- `excel_path` (str): Path to the Excel file.
- `folder_name` (str or bool, default=False): Folder name for the sheet name, or use the file name.

Functionality
1. Converts `filtered_ranges` to a string and creates a DataFrame with results.
2. Determines the sheet name (based on `folder_name` or `file_name`) and limits it to 31 characters.
3. Ensures the directory for `excel_path` exists.
4. Writes data to the Excel file:
   - Appends to an existing sheet or creates a new one.
   - Creates a new file if it doesn't exist.

Output
- Saves results to the specified Excel file and prints a confirmation message.
'''

def write_to_excel(file_name, restfreq, rms, filtered_ranges, channel_ranges_str, excel_path, folder_name=False):
    # Write the results to an Excel file, sheet name based on file or folder name
    
    # convert filtered_ranges to string
    channel_ranges_str = '; '.join([f"{start}-{end}" for start, end in filtered_ranges])
    output_data = {
        "filename": [file_name],
        "time": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "restfreq_Hz": [restfreq],
        "rms": [rms],
        "moment_channels": [channel_ranges_str],
    }
    df = pd.DataFrame(output_data)

    # worksheet name based on file or folder name
    if folder_name and os.path.dirname(file_name):
        sheet_name = os.path.basename(os.path.dirname(file_name))
    else:
        sheet_name = os.path.splitext(os.path.basename(file_name))[0]
    # limit sheet name length to 31 characters
    sheet_name = sheet_name[:31]

    # check if the directory exists, create it if not
    
    if os.path.dirname(excel_path):
        os.makedirs(os.path.dirname(excel_path), exist_ok=True)

    # if Excel file already exists, read existing data and append
    if os.path.exists(excel_path):
        with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a" ,if_sheet_exists="overlay") as writer:
            try:
                existing_df = pd.read_excel(excel_path, sheet_name=sheet_name)
                df = pd.concat([existing_df, df], ignore_index=True)
            except ValueError:
                # if sheet does not exist, write directly
                pass
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        # if the file does not exist, create a new one
        with pd.ExcelWriter(excel_path, engine="openpyxl", mode="w") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Write in Excel: {excel_path}，Sheet: {sheet_name}")