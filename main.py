'''
Main Program Workflow  

Adjustable Parameters  
- `window`: Center region size.  
- `sigma`: Gaussian smoothing standard deviation.  
- `thresh_ratio`: Slope threshold ratio.  
- `fraction`: Edge calculation denominator.  
- `number_sigma`: Signal mask threshold multiplier.  
- `excel_path`: Path to save results in Excel.  
- `min_length`: Minimum length of emission ranges.  

Workflow 
1. File Selection: Choose a single file (`f`) or a folder (`d`).  
2. Read Data: Extract spectrum and cube information.  
3. Smooth Spectrum: Apply Gaussian smoothing and calculate slope.  
4. Detect Ranges: Identify emission ranges based on slope.  
5. Filter Ranges: Exclude invalid ranges and plot results.  
6. Build Masks: Create RMS and signal masks.  
7. Moment Maps: Generate and save moment maps.  
8. Save Results: Write results to an Excel file.  
'''
from utils import (
    get_center_spectrum,
    smooth_and_slope,
    find_emission_ranges,
    filter_ranges,
    create_rms_mask,
    create_signal_mask,
    calculate_moments,
    write_to_excel,
    select_file,
    select_folder_and_find_fits
)
# main
# ===== Adjustable Parameters =====
window = 5            # Center region size
sigma = 1.5           # Gaussian smoothing standard deviation
thresh_ratio = 0.1    # Slope threshold ratio
fraction = 8          # Edge calculation denominator
number_sigma = 3      # Signal mask threshold multiplier
excel_path = "moment_info.xlsx"  # Excel file path
min_length = 5      # Minimum length of emission ranges


mode = input("Enter ‘f’ to select manually, or ‘d’ to select folder and automatically search for FITS files:")
if mode == 'f':
    file_list = [select_file()]
    folder_name = None
elif mode == 'd':
    file_list, folder_name = select_folder_and_find_fits()
else:
    raise ValueError("Please enter ‘f’ or 'd'.")

print("Files used:")

for file_name in file_list:
    print(file_name)
    # ===== Main Program =====
    # 1. Read cube and collect center spectrum
    spectrum, data, n_chan, ny, nx, cube, base, restfreq = get_center_spectrum(file_name, window=window)

    # 2. Smooth the spectrum and calculate the slope
    spectrum_smooth, slope, threshold_dy = smooth_and_slope(spectrum, sigma=sigma, thresh_ratio=thresh_ratio)

    # 3. Detect signal ranges
    keep_ranges = find_emission_ranges(slope, threshold_dy, min_length=min_length)

    # 4. Filter signal ranges, plot figures
    filtered_ranges, restfreq, channel_ranges_str = filter_ranges(
        keep_ranges, n_chan, spectrum, spectrum_smooth, slope, threshold_dy, file_name, fraction=fraction, base=base
    )

    # 5. Build RMS mask
    edge = n_chan // fraction
    noise_mask, rms = create_rms_mask(data, filtered_ranges, n_chan, edge)

    # 6. Build signal mask
    cube_mask = create_signal_mask(data, filtered_ranges, rms, number_sigma=number_sigma)

    # 7. Do the moment map and save
    cube_masked = cube.with_mask(cube_mask)
    calculate_moments(cube_masked, base)

    # 8. Write to Excel
    write_to_excel(file_name, restfreq, rms, filtered_ranges, channel_ranges_str, excel_path=excel_path, folder_name = folder_name)