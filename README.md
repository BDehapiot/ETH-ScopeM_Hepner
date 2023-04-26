# ETH-ScopeM_Hepner
Extract numerical data from bmp image plots 

## Request
The goal here is to extract all numerical information contained in bmp image plots (e.g. `data/expl_01.bmp`) to allow further analysis. 

## Installation
1 - Download this GitHub repository  

2 - Install miniconda:  
https://docs.conda.io/en/latest/miniconda.html  

3 - Run conda prompt and install mamba in base conda environment:  
`conda install mamba -n base -c conda-forge`  

4 - Run conda prompt, from downloaded repository, and install conda environment:  
`mamba env create -f environment.yml`   

5 - Activate conda environment:  
`conda activate ETH-ScopeM_Hepner`  

6 - (Optional) Install spyder IDE:  
`pip install spyder` 

## Outputs
Plot features are extracted as follow:
- `cmap.tif` colormap (log scale) saved as a `uint8` tif image 
- `cmap.csv` colormap (log scale) saved as a comma separated csv file 
- `cmap-raw.tif` colormap (real values) saved as a `float32` tif image 
- `cmap-raw.csv` colormap (real values) saved as a comma separated csv file 
- `line.tif` line surrounding the colormap saved as a `uint8` tif image
- `dot-labels.tif` labeled dots surrounding the colormap saved as a `uint8` tif image
- `dot-info.csv` dot xy coordinates saved as a comma separated csv file 
- `plot.png` extracted features plot for sanity check saved as png image


