#%% Imports -------------------------------------------------------------------

import numpy as np
import pandas as pd
from skimage import io 
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist
from skimage.segmentation import expand_labels
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt
from skimage.metrics import structural_similarity
from scipy.cluster.hierarchy import fcluster, linkage
from skimage.morphology import (
    disk, remove_small_objects, binary_closing, erosion, dilation, flood_fill
    )

#%% Initialize ----------------------------------------------------------------

img_name = 'expl_03.bmp'

# Open data
img = io.imread(Path('data', img_name))
dLib1 = io.imread(Path('data/lib', 'dLib1.tif'))
dLib2 = io.imread(Path('data/lib', 'dLib2.tif'))

# Get avg & std img (RGB channels)
avg_img = np.mean(img, axis=2).astype('uint8')
std_img = np.std(img, axis=2)

#%% Cmap & line ---------------------------------------------------------------

# Extract img_plot
mask = std_img
mask[mask!=0] = 255
cmap_img = avg_img.copy()
cmap_img[mask==255] = 255
mask = cmap_img != 255
mask = remove_small_objects(mask, min_size=8192)
cmap_img[mask==0] = 0
cmap_img = cmap_img.astype('uint8')

# Clean img_plot
for labl in np.unique(cmap_img):
    for prop in regionprops(label(cmap_img == labl)):
        if prop.area / prop.perimeter_crofton < 0.75:
            cmap_img[prop.coords[:,0], prop.coords[:,1]] = 0

# Extract img_plot 
line_img = cmap_img > 0
line_img = binary_closing(line_img, footprint=disk(15))
line_img = line_img ^ erosion(line_img)

# Fill empty gaps (within line)
cmap_img = expand_labels(cmap_img, distance=3)
mask = flood_fill(line_img, (0, 0), 255, connectivity=1)
cmap_img[mask == True] = 0

# # Display
# import napari
# viewer = napari.Viewer()
# viewer.add_image(line_img)
# viewer.add_image(cmap_img, blending='additive')

#%% Dots & associated numbers -------------------------------------------------

# Extract dot_mask (using blue channel)
dot_mask = img[...,2] == 0

# Read numbers
dot_data = []
for prop in regionprops(label(dot_mask)):
    
    ctrd_y = round(prop.centroid[0])
    ctrd_x = round(prop.centroid[1])
    crop = avg_img[ctrd_y-6:ctrd_y+6,ctrd_x-6:ctrd_x+6]
    circularity = round(4 * np.pi * prop.area / (prop.perimeter ** 2), 3)
    isDigit = False if circularity > 1 else True

    if isDigit:
        ssim = []
        for d in dLib2:
            ssim.append(structural_similarity(crop, d))
        digit = np.argmax(np.stack(ssim))
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

# Get object clusters 
ctrd_y = [data[0] for data in dot_data]
ctrd_x = [data[1] for data in dot_data]
digit = [data[5] for data in dot_data]
distances = pdist(list(zip(ctrd_x, ctrd_y)))
linkage_matrix = linkage(distances, method='single')
cluster_label = fcluster(linkage_matrix, 50, criterion='distance')

# Update dot_data with cluster labels
for i, labl in enumerate(cluster_label):   
    dot_data[i].append(labl)
    
# Update dot_data with numbers
for labl in np.unique(cluster_label):
    
    idx = np.where(cluster_label==labl)
    
    if len(idx[0]) == 2:        
        d = np.array(digit)[idx]
        number = int(d[~np.isnan(d)])

    if len(idx[0]) > 2:    
        d = np.array(digit)[idx]
        nan_idx = np.where(np.isnan(d))
        y = np.array(ctrd_y)[idx]
        x = np.array(ctrd_x)[idx]
        d = np.delete(d, nan_idx) 
        y = np.delete(y, nan_idx) 
        x = np.delete(x, nan_idx) 
        d1 = str(int(d[np.argmin(x)]))
        d2 = str(int(d[np.argmax(x)]))
        number = int(d1 + d2)

    for i, data in enumerate(dot_data):            
        if data[6] == labl and data[4] == False:
            dot_data[i][5] = number
            
# Extract dot_img
dot_img_labels = np.zeros_like(cmap_img)
for i, data in enumerate(dot_data):
    if not data[4]:
        edm = np.zeros_like(cmap_img, dtype=float)
        edm[data[0], data[1]] = 1
        edm = distance_transform_edt(1 - edm)
        edm[line_img == 0] = 255       
        y, x = np.unravel_index(np.argmin(edm), edm.shape)
        dot_data[i][0] = y; dot_data[i][1] = x; 
        dot_img_labels[y, x] = data[5]

dot_img = dot_img_labels > 0
dot_img_labels = dilation(dot_img_labels, footprint=disk(5))

# # Display
# import napari
# viewer = napari.Viewer()
# viewer.add_image(dot_img)

#%% Grayscale & associated numbers --------------------------------------------

# Extract gray scale (gScale)
gScale = np.linspace(avg_img[7,1], avg_img[7,-1], avg_img.shape[1]-2)

# Extract number scale (nScale_img)
nScale_img = avg_img[17:24,1:-1].astype('uint8')
nScale_mask = np.invert(nScale_img) > 10
nScale_mask = remove_small_objects(nScale_mask, min_size=4)

# Read numbers
nScale_data = []
for prop in regionprops(label(nScale_mask)):

    ctrd_y = round(prop.centroid[0])
    ctrd_x = round(prop.centroid[1])    
    min_x = np.min(prop.coords[:,1])
    max_x = np.max(prop.coords[:,1])
    crop = nScale_img[:,min_x:max_x+1]
    width = max_x - min_x
    nDigits = round(width/5)

    number = ''    
    for i in range(nDigits):       
        ccrop = crop[:,i*5:i*5+5]
        
        ssim = []
        for d in dLib1:            
            ssim.append(structural_similarity(ccrop, d, win_size=5))
        number = number + str(np.argmax(np.stack(ssim)))
    
    nScale_data.append([
        ctrd_y,
        ctrd_x,
        crop,
        nDigits,
        int(number),
        ])

# Exponential function
def exponential(x, a, b, c):
    return a * np.exp(b * x) + c

# Extract x and y (ctrd_x and number)
x = [data[1] for data in nScale_data]
y = [data[4] for data in nScale_data]

# Estimate initial parameters
a0 = y[0]
b0 = np.log(y[-1]/y[0])/(x[-1] - x[0])
c0 = np.min(y)

# Fit data
popt, pcov = curve_fit(exponential, x, y, p0=[a0, b0, c0])
x_fit = np.linspace(0, avg_img.shape[1]-2, avg_img.shape[1]-2)
nScale = exponential(x_fit, *popt)

# Convert cmap_img to linear scale 
cmap_img_raw = np.zeros_like(cmap_img, dtype='float') 
for gInt in np.unique(cmap_img):
    if gInt != 0:
        nInt = nScale[np.argmin(np.abs(gScale-gInt))]
        coords = np.where(cmap_img == gInt)
        cmap_img_raw[coords] = nInt

# # Plot
# plt.scatter(x, y, label='Data')
# plt.plot(x_fit, nScale, label='Exponential fit')
# plt.legend()
# plt.show()

#%% Output: data

dot_info = []       
for lab in np.unique(dot_img_labels):  
    if lab > 0:
        y, x = np.where(dot_img_labels == lab)
        dot_info.append((lab, int(np.mean(y)), int(np.mean(x))))
dot_info = pd.DataFrame(
    np.stack(dot_info), 
    columns=['#dot', 'y-coord', 'x-coord'], 
    )


np.savetxt(
    Path('data', img_name.replace('.bmp', '_dot-info.csv')),
    dot_info, delimiter=',', fmt='%d'
    )

np.savetxt(
    Path('data', img_name.replace('.bmp', '_cmap.csv')),
    cmap_img, delimiter=',', fmt='%d'
    )

np.savetxt(
    Path('data', img_name.replace('.bmp', '_cmap-raw.csv')),
    cmap_img_raw, delimiter=',', fmt='%.3f'
    )

#%% Output: images
        
io.imsave(
    Path('data', img_name.replace('.bmp', '_cmap.tif')),
    cmap_img, check_contrast=False,
    )

io.imsave(
    Path('data', img_name.replace('.bmp', '_cmap-raw.tif')),
    cmap_img_raw.astype('float32'), check_contrast=False,
    )

io.imsave(
    Path('data', img_name.replace('.bmp', '_dot-labels.tif')),
    dot_img_labels, check_contrast=False,
    ) 

io.imsave(
    Path('data', img_name.replace('.bmp', '_line.tif')),
    line_img*255, check_contrast=False,
    )  

# # Display
# import napari
# viewer = napari.Viewer()
# viewer.add_image(line_img)
# viewer.add_image(dot_img_labels > 0, blending='additive')
# viewer.add_image(cmap_img, blending='additive')

#%% Output: plot

dpi = 300
plotSize = 0.6
linewidth = 0.5
fontSize = 8

# -----------------------------------------------------------------------------

rcParams['axes.linewidth'] = linewidth
rcParams['axes.titlesize'] = fontSize * 1.5
rcParams['axes.labelsize'] = fontSize * 0.75
rcParams['xtick.major.width'] = linewidth
rcParams['ytick.major.width'] = linewidth
rcParams['xtick.labelsize'] = fontSize * 0.6
rcParams['ytick.labelsize'] = fontSize * 0.6

# -----------------------------------------------------------------------------

# Set figure layout
width = img.shape[1]
height = img.shape[0]
fig_width = width / dpi
fig_height = height / dpi
fig_width /= plotSize
fig_height /= plotSize
bottom = (1 - plotSize) * 0.5
top = bottom + plotSize
left = (1 - plotSize) * 0.5
right = left + plotSize

# Prepare data for plot
plot_y, plot_x = np.nonzero(cmap_img)
plot_ctrd_y = round(np.mean(plot_y))
plot_ctrd_x = round(np.mean(plot_x))
dot_y, dot_x = np.nonzero(dot_img)
dot_labels = dot_img_labels[dot_y, dot_x]
cmap_img_mask = np.ma.masked_where(cmap_img == 0, cmap_img)

# Plot
fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
fig.subplots_adjust(top=top, bottom=bottom, right=right, left=left)
plt.ylim([img.shape[0], 0])
plt.xlim([0, img.shape[1]])
ax.set_xlabel('x position (pixel)')
ax.set_ylabel('y position (pixel)')

# -----------------------------------------------------------------------------

plot1 = plt.imshow(
    np.invert(cmap_img_mask), 
    cmap='plasma_r', 
    vmin=np.min(gScale), 
    vmax=np.max(gScale)
    )

# -----------------------------------------------------------------------------

plot2 = plt.scatter(dot_x, dot_y, s=10, c='red')
for i, labl in enumerate(dot_labels):
    
    y = dot_y[i]; x = dot_x[i]
    cy = plot_ctrd_y; cx = plot_ctrd_x
    edm_dot = np.zeros_like(cmap_img, dtype=float)
    edm_ctrd = np.zeros_like(cmap_img, dtype=float)
    edm_dot[y, x] = 1; edm_ctrd[cy, cx] = 1
    edm_dot = distance_transform_edt(1 - edm_dot)    
    edm_ctrd = distance_transform_edt(1 - edm_ctrd) 
    edm_ctrd[edm_dot > 25] = 0
    txt_y, txt_x = np.unravel_index(np.argmax(edm_ctrd), edm_ctrd.shape)
    plt.text(
        txt_x, txt_y, labl, 
        fontsize=fontSize * 0.75,
        horizontalalignment='center', 
        verticalalignment='center'
        )
    
# -----------------------------------------------------------------------------

cbax = fig.add_axes([left, top, plotSize, 0.025])
cbar = plt.colorbar(plot1, orientation='horizontal', cax=cbax)
cbax.set_xlabel('???')
cbax.xaxis.set_ticks_position('top')
cbax.xaxis.set_label_position('top')
cbar.set_ticks(np.linspace(np.min(gScale), np.max(gScale), 6))
cbar.set_ticklabels([data[4] for data in nScale_data])

# -----------------------------------------------------------------------------

# Save figure
plt.savefig("output.tif", dpi=dpi)

#%% Output: images
       
# # Display
# import napari
# viewer = napari.Viewer()
# viewer.add_image(cmap_img_raw)

# io.imsave(
#     Path('data', img_name.replace('.bmp', '_plot.tif')),
#     cmap_img, check_contrast=False,
#     )
# io.imsave(
#     Path('data', img_name.replace('.bmp', '_dots.tif')),
#     dot_img*255, check_contrast=False,
#     ) 
# io.imsave(
#     Path('data', img_name.replace('.bmp', '_line.tif')),
#     line_img, check_contrast=False,
#     )  
