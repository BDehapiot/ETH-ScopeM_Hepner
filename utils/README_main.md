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