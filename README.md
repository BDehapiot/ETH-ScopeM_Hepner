![Python Badge](https://img.shields.io/badge/Python-3.9-rgb(69%2C132%2C182)?logo=python&logoColor=rgb(149%2C157%2C165)&labelColor=rgb(50%2C60%2C65))  
![Author Badge](https://img.shields.io/badge/Author-Benoit%20Dehapiot-blue?labelColor=rgb(50%2C60%2C65)&color=rgb(149%2C157%2C165))
![Date Badge](https://img.shields.io/badge/Created-2023--03--28-blue?labelColor=rgb(50%2C60%2C65)&color=rgb(149%2C157%2C165))
![License Badge](https://img.shields.io/badge/Licence-GNU%20General%20Public%20License%20v3.0-blue?labelColor=rgb(50%2C60%2C65)&color=rgb(149%2C157%2C165))     

# ETH-ScopeM_Hepner  
Extract numerical data from bmp image plots

## Index
- [Installation](#installation)
- [Outputs](#outputs)
- [Comments](#comments)

## Installation

Pease select your operating system

<details> <summary>Windows</summary>  

### Step 1: Download this GitHub Repository 
- Click on the green `<> Code` button and download `ZIP` 
- Unzip the downloaded file to a desired location

### Step 2: Install Miniforge (Minimal Conda installer)
- Download and install [Miniforge](https://github.com/conda-forge/miniforge) for your operating system   
- Run the downloaded `.exe` file  
    - Select "Add Miniforge3 to PATH environment variable"  

### Step 3: Setup Conda 
- Open the newly installed Miniforge Prompt  
- Move to the downloaded GitHub repository
- Run the following command:  
```bash
mamba env create -f environment.yml
```
- Activate Conda environment:
```bash
conda activate Hepner
```
Your prompt should now start with `(Hepner)` instead of `(base)`

</details> 

<details> <summary>MacOS</summary>  

### Step 1: Download this GitHub Repository 
- Click on the green `<> Code` button and download `ZIP` 
- Unzip the downloaded file to a desired location

### Step 2: Install Miniforge (Minimal Conda installer)
- Download and install [Miniforge](https://github.com/conda-forge/miniforge) for your operating system   
- Open your terminal
- Move to the directory containing the Miniforge installer
- Run one of the following command:  
```bash
# Intel-Series
bash Miniforge3-MacOSX-x86_64.sh
# M-Series
bash Miniforge3-MacOSX-arm64.sh
```   

### Step 3: Setup Conda 
- Re-open your terminal 
- Move to the downloaded GitHub repository
- Run the following command: 
```bash
mamba env create -f environment.yml
```  
- Activate Conda environment:  
```bash
conda activate Hepner
```
Your prompt should now start with `(Hepner)` instead of `(base)`

</details>

## Outputs
Plot features are extracted as follow:
```bash
- cmap.tif             # colormap (log scale) saved as a `uint8` tif image 
- cmap.csv             # colormap (log scale) saved as a comma separated csv file 
- cmap-raw.tif         # colormap (real values) saved as a `float32` tif image 
- cmap-raw.csv         # colormap (real values) saved as a comma separated csv file 
- line.tif line        # surrounding the colormap saved as a `uint8` tif image
- dot-labels.tif       # labeled dots surrounding the colormap saved as a `uint8` tif image
- dot-info.csv         # dot xy coordinates saved as a comma separated csv file 
- plot.png extracted   # features plot for sanity check saved as png image
```

## Comments