# ETH-ScopeM_Hepner
Extract numerical data from bmp image plots 

## Request
The goal here is to extract all numerical information contained in bmp image plots (e.g. `data/expl_01.bmp`) to allow further analysis. 

## Outputs
Plot features are extracted as follow:
- `cmap.tif` colormap (log scale) saved as a `uint8` tif image 
- `cmap.csv` colormap (log scale) saved as a comma separated csv file 
- `cmap-raw.tif` colormap (real values) saved as a `float32` tif image 
- `cmap-raw.csv` colormap (real values) saved as a comma separated csv file 
- `line.tif` line surrounding the colormap saved as a `uint8` tif image
- `dot-labels.tif` labeled dots surrounding the colormap saved as a `uint8` tif image
- `dot-info.csv` dot xy coordinates saved as a comma separated csv file 


