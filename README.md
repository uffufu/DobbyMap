# DOBBYmap

DOBBY DO YOUR SPECTRAL WORK!

Automated spectral data processing tool that supports single or batch FITS cubes, generates moment map FITS files, and outputs analysis results to Excel.

This project extracts spectral data from the centre of a data cube and automatically detects potential emission signal ranges based on the first derivative of the spectral intensity. The identified signal channels are then used to generate moment maps of order 0 to 2.

## Features 
- Support single FITS cube or batch processing of all FITS cubes in a folder
- Detect frequency range automatically and estimate RMS
- Generate moment 0/1/2 FITS file
- RMS and channel used written into Excel（one sheet for each file）

## Installation

1. Python 3.9（or compatible version）recommended
2. install required packages：
   ```bash
   pip install -r requirements.txt
   ```

## Instructions

### 1. Run main.py directly
```bash
python main.py
```
Or Directly Run All Cells of Jupyter Notebook `main.ipynb` or `DobbyMap.ipynb`. 

### 2. Mode selection
- enter `f`：for select single FITS cube manually
- enter `d`：select a folder, automatically process all FITS cubes in the folder

### 3. Output
- moment map is saved to the execution directory (or specified folder)
- Statistical data is saved to `moment_info.xlsx`

## Structure

```
DOBBYmap/
├── DobbyMap.ipynb    # Include main program and tool functions in one Jupyter Notebook
├── main.py           # Main program
├── utils.py          # Tool function
├── requirements.txt  # Packages list
└──  README.md         # Instruction manual
```

## Acknowledgement

Part of the programming and documentation for this project was completed with the help of AI tools.
