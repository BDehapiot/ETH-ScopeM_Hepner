#%% Imports

import numpy as np
from skimage import io 
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.morphology import disk
from skimage.filters.rank import minimum
from scipy.spatial.distance import pdist
from skimage.measure import label, regionprops
from skimage.metrics import structural_similarity
from scipy.cluster.hierarchy import fcluster, linkage

#%% Initialize

img_name = 'expl_01.bmp'
digits_name = 'digits.tif'
img = io.imread(Path('data', img_name))
digits = io.imread(Path('data', digits_name))

#%%

img_avg = np.mean(img, axis=2)
img_std = np.std(img, axis=2)

# -----------------------------------------------------------------------------

img_clean = img_avg.copy()
img_clean[img_std!=0] = 255
img_clean[img_clean==255] = 0
img_clean = minimum(img_clean.astype('uint8'), disk(3))

# -----------------------------------------------------------------------------

# Get circularity function
def circ(area, perimeter):
    return 4 * np.pi * area / (perimeter ** 2)

dot_data = []
dot_digits = []
dot_mask = img[...,2] == 0
dot_props = regionprops(label(dot_mask))
for prop in dot_props:
    
    y_ctrd = round(prop.centroid[0])
    x_ctrd = round(prop.centroid[1])
    crop = img[...,2][y_ctrd-6:y_ctrd+6,x_ctrd-6:x_ctrd+6]
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
        y_ctrd, 
        x_ctrd,
        crop,
        circularity,
        isDigit,
        digit,
        ])
    
# -----------------------------------------------------------------------------

# Cluster object according to max_distance
max_distance = 50
y_ctrd = [data[0] for data in dot_data]
x_ctrd = [data[1] for data in dot_data]
digit = [data[5] for data in dot_data]
distances = pdist(list(zip(x_ctrd, y_ctrd)))
linkage_matrix = linkage(distances, method='single')
cluster_label = fcluster(linkage_matrix, max_distance, criterion='distance')

# Update dot_data with cluster labels
for i, labl in enumerate(cluster_label):   
    dot_data[i].append(labl)
    
# -----------------------------------------------------------------------------    

# Update dot_data with dot-associated numbers
for labl in np.unique(cluster_label):
    
    idx = np.where(cluster_label==labl)
    
    if len(idx[0]) == 2:        
        d = np.array(digit)[idx]
        number = int(d[~np.isnan(d)])

    if len(idx[0]) > 2:    
        d = np.array(digit)[idx]
        nan_idx = np.where(np.isnan(d))
        y = np.array(y_ctrd)[idx]
        x = np.array(x_ctrd)[idx]
        d = np.delete(d, nan_idx) 
        y = np.delete(y, nan_idx) 
        x = np.delete(x, nan_idx) 
        d1 = str(int(d[np.argmin(x)]))
        d2 = str(int(d[np.argmax(x)]))
        number = int(d1 + d2)

    for i, data in enumerate(dot_data):            
        if data[6] == labl and data[4] == False:
            dot_data[i][5] = number

#%% Display

# import napari
# viewer = napari.Viewer()
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
# plt.imshow(img_avg, cmap='gray')
# plt.imshow(img_std, cmap='gray')
# plt.imshow(img_clean, cmap='gray')
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
#     img_clean.astype('uint8'),
#     check_contrast=False,
#     )

# io.imsave(
#     Path('data', img_name.replace('.bmp', '_dots_labels.tif')),
#     dot_labels.astype('uint8'),
#     check_contrast=False,
#     )
