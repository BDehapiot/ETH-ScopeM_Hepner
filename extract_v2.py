#%% Imports -------------------------------------------------------------------

import numpy as np
from skimage import io 
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from skimage.filters.rank import minimum
from scipy.spatial.distance import pdist
from skimage.segmentation import expand_labels
from skimage.measure import label, regionprops
from skimage.metrics import structural_similarity
from scipy.cluster.hierarchy import fcluster, linkage
from skimage.morphology import (
    disk, remove_small_objects, binary_closing, erosion, flood_fill
    )

#%% Initialize ----------------------------------------------------------------

img_name = 'expl_06.bmp'

# Open data
img = io.imread(Path('data', img_name))
dLib1 = io.imread(Path('data/lib', 'dLib1.tif'))
dLib2 = io.imread(Path('data/lib', 'dLib2.tif'))

# Get avg & std img (RGB channels)
avg_img = np.mean(img, axis=2)
std_img = np.std(img, axis=2)

#%% Extract plot & line -------------------------------------------------------

# Extract img_plot
mask = std_img
mask[mask!=0] = 255
plot_img = avg_img.copy()
plot_img[mask==255] = 255
mask = plot_img != 255
mask = remove_small_objects(mask, min_size=8192)
plot_img[mask==0] = 0
plot_img = plot_img.astype('uint8')

# Clean img_plot
for labl in np.unique(plot_img):
    for prop in regionprops(label(plot_img == labl)):
        if prop.area / prop.perimeter_crofton < 0.75:
            plot_img[prop.coords[:,0], prop.coords[:,1]] = 0
plot_img = expand_labels(plot_img, distance=3)

# Extract img_plot 
img_line = plot_img > 0
img_line = binary_closing(img_line, footprint=disk(15))
img_line = img_line ^ erosion(img_line)

# Remove labels outside of img_line
mask = flood_fill(img_line, (0, 0), 255, connectivity=1)
plot_img[mask == True] = 0

# # Display
# import napari
# viewer = napari.Viewer()
# viewer.add_image(img_line)
# viewer.add_image(plot_img, blending='additive')

#%% Extract dots & associated numbers -----------------------------------------

# Circularity function
def circ(area, perimeter):
    return 4 * np.pi * area / (perimeter ** 2)

dot_data = []
dot_digits = []
dot_mask = img[...,2] == 0
dot_props = regionprops(label(dot_mask))
for prop in dot_props:
    
    ctrd_y = round(prop.centroid[0])
    ctrd_x = round(prop.centroid[1])
    crop = img[...,2][ctrd_y-6:ctrd_y+6,ctrd_x-6:ctrd_x+6]
    circularity = round(circ(prop.area, prop.perimeter), 3)
    isDigit = False if circularity > 1 else True

    if isDigit:
        ssim = []
        for digit in digits:
            ssim.append(structural_similarity(crop, digit))
        digit = np.argmax(np.stack(ssim))
        dot_digits.append(crop)
    else:
        digit = np.nan

    dot_data.append([
        ctrd_y, 
        ctrd_x,
        crop,
        circularity,
        isDigit,
        digit,
        ])


