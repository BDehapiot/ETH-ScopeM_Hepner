#%% Imports

import numpy as np
from skimage import io 
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from skimage.filters.rank import minimum
from scipy.spatial.distance import pdist
from skimage.measure import label, regionprops
from skimage.metrics import structural_similarity
from scipy.cluster.hierarchy import fcluster, linkage
from skimage.morphology import (
    disk, remove_small_objects, binary_closing, erosion, flood_fill
    )

#%% Functions

# Circularity function
def circ(area, perimeter):
    return 4 * np.pi * area / (perimeter ** 2)

#%% Initialize

img_name = 'expl_06.bmp'

# Open data
img = io.imread(Path('data', img_name))
dLib1 = io.imread(Path('data/lib', 'dLib1.tif'))
dLib2 = io.imread(Path('data/lib', 'dLib2.tif'))

# Get avg and std img
avg_img = np.mean(img, axis=2)
std_img = np.std(img, axis=2)

#%% Get plot_img

from skimage.morphology import binary_closing, erosion, flood_fill
from skimage.segmentation import expand_labels

mask = std_img.copy()
mask[mask!=0] = 255
plot_img = avg_img.copy()
plot_img[mask==255] = 255
mask = plot_img != 255
mask = remove_small_objects(mask, min_size=8192)
plot_img[mask==0] = 0
plot_img = plot_img.astype('uint8')

# -----------------------------------------------------------------------------

img_line = plot_img > 0
img_line = binary_closing(img_line, footprint=disk(15))
img_line = img_line ^ erosion(img_line)

# -----------------------------------------------------------------------------

for labl in np.unique(plot_img):
    for prop in regionprops(label(plot_img == labl)):
        if prop.area / prop.perimeter_crofton < 0.75:
            plot_img[prop.coords[:,0], prop.coords[:,1]] = 0

# -----------------------------------------------------------------------------

mask = flood_fill(img_line, (0, 0), 255, connectivity=1)
plot_img = expand_labels(plot_img, distance=3)
plot_img[mask == True] = 0

# -----------------------------------------------------------------------------

# Display
import napari
viewer = napari.Viewer()
# viewer.add_image(plot_img == 128)
# viewer.add_image(test)
# viewer.add_image(mask)
viewer.add_image(img_line)
viewer.add_image(plot_img, blending='additive')

#%% Get grayscale

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

# # -----------------------------------------------------------------------------

# # Exponential function
# def exponential(x, a, b, c):
#     return a * np.exp(b * x) + c

# # Extract x and y (ctrd_x and number)
# x = [data[1] for data in nScale_data]
# y = [data[4] for data in nScale_data]

# # Estimate initial parameters
# a0 = y[0]
# b0 = np.log(y[-1]/y[0])/(x[-1] - x[0])
# c0 = np.min(y)

# # Fit data
# popt, pcov = curve_fit(exponential, x, y, p0=[a0, b0, c0])
# x_fit = np.linspace(0, avg_img.shape[1]-2, avg_img.shape[1]-2)
# nScale = exponential(x_fit, *popt)

# # Plot
# plt.scatter(x, y, label='Data')
# plt.plot(x_fit, nScale, label='Exponential fit')
# plt.legend()
# plt.show()

#%% Get img_dot

# # Circularity function
# def circ(area, perimeter):
#     return 4 * np.pi * area / (perimeter ** 2)

# dot_data = []
# dot_digits = []
# dot_mask = img[...,2] == 0
# dot_props = regionprops(label(dot_mask))
# for prop in dot_props:
    
#     ctrd_y = round(prop.centroid[0])
#     ctrd_x = round(prop.centroid[1])
#     crop = img[...,2][ctrd_y-6:ctrd_y+6,ctrd_x-6:ctrd_x+6]
#     circularity = round(circ(prop.area, prop.perimeter), 3)
#     isDigit = False if circularity > 1 else True

#     if isDigit:
#         ssim = []
#         for digit in digits:
#             ssim.append(structural_similarity(crop, digit))
#         digit = np.argmax(np.stack(ssim))
#         dot_digits.append(crop)
#     else:
#         digit = np.nan

#     dot_data.append([
#         ctrd_y, 
#         ctrd_x,
#         crop,
#         circularity,
#         isDigit,
#         digit,
#         ])
    
# # -----------------------------------------------------------------------------

# # Cluster object according to max_distance
# max_distance = 50
# ctrd_y = [data[0] for data in dot_data]
# ctrd_x = [data[1] for data in dot_data]
# digit = [data[5] for data in dot_data]
# distances = pdist(list(zip(ctrd_x, ctrd_y)))
# linkage_matrix = linkage(distances, method='single')
# cluster_label = fcluster(linkage_matrix, max_distance, criterion='distance')

# # Update dot_data with cluster labels
# for i, labl in enumerate(cluster_label):   
#     dot_data[i].append(labl)
    
# # -----------------------------------------------------------------------------    

# # Update dot_data with dot-associated numbers
# for labl in np.unique(cluster_label):
    
#     idx = np.where(cluster_label==labl)
    
#     if len(idx[0]) == 2:        
#         d = np.array(digit)[idx]
#         number = int(d[~np.isnan(d)])

#     if len(idx[0]) > 2:    
#         d = np.array(digit)[idx]
#         nan_idx = np.where(np.isnan(d))
#         y = np.array(ctrd_y)[idx]
#         x = np.array(ctrd_x)[idx]
#         d = np.delete(d, nan_idx) 
#         y = np.delete(y, nan_idx) 
#         x = np.delete(x, nan_idx) 
#         d1 = str(int(d[np.argmin(x)]))
#         d2 = str(int(d[np.argmax(x)]))
#         number = int(d1 + d2)

#     for i, data in enumerate(dot_data):            
#         if data[6] == labl and data[4] == False:
#             dot_data[i][5] = number

#%% Display

# import napari
# viewer = napari.Viewer()
# viewer.add_image(plot_img)
# viewer.add_image(digits)
# viewer.add_image(np.stack(dot_digits))

#%%

# # Get digits array (works with expl_01.bmp)
# digits = np.zeros((10, 12, 12))
# digits[0,...] = dot_digits[10] # 0
# digits[1,...] = dot_digits[0] # 1
# digits[2,...] = dot_digits[1] # 2
# digits[3,...] = dot_digits[4] # 3
# digits[4,...] = dot_digits[5] # 4
# digits[5,...] = dot_digits[8] # 5
# digits[6,...] = dot_digits[11] # 6
# digits[7,...] = dot_digits[14] # 7
# digits[8,...] = dot_digits[13] # 8
# digits[9,...] = dot_digits[12] # 9

# io.imsave(
#     Path('data', 'digits.tif'),
#     digits.astype('uint8'),
#     check_contrast=False,
#     )

#%% Plot

# fig, ax = plt.subplots() 
# plt.imshow(img, cmap='gray')
# plt.imshow(img[...,0], cmap='gray') # Red
# plt.imshow(img[...,1], cmap='gray') # Green
# plt.imshow(img[...,2], cmap='gray') # Blue
# plt.imshow(avg_img, cmap='gray')
# plt.imshow(std_img, cmap='gray')
# plt.imshow(plot_img, cmap='gray')
# plt.imshow(dot_mask, cmap='gray')
# plt.imshow(dot_labels, cmap='gray')
# ax.set_axis_off()

#%% Save

# io.imsave(
#     Path('data', img_name.replace('.bmp', '_red.tif')),
#     img[...,0].astype('uint8'),
#     check_contrast=False,
#     )

# io.imsave(
#     Path('data', img_name.replace('.bmp', '_green.tif')),
#     img[...,1].astype('uint8'),
#     check_contrast=False,
#     )

# io.imsave(
#     Path('data', img_name.replace('.bmp', '_blue.tif')),
#     img[...,2].astype('uint8'),
#     check_contrast=False,
#     )

# io.imsave(
#     Path('data', img_name.replace('.bmp', '_clean.tif')),
#     plot_img.astype('uint8'),
#     check_contrast=False,
#     )

# io.imsave(
#     Path('data', img_name.replace('.bmp', '_dots_labels.tif')),
#     dot_labels.astype('uint8'),
#     check_contrast=False,
#     )
