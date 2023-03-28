#%% Imports

import numpy as np
from skimage import io 
from pathlib import Path
import matplotlib.pyplot as plt

#%% Initialize

img_name = 'ert_sw.bmp'
img = io.imread(Path('data', img_name))

#%%

img_avg = np.mean(img, axis=2)
img_std = np.std(img, axis=2)

# -----------------------------------------------------------------------------

from skimage.morphology import disk
from skimage.filters.rank import minimum

img_clean = img_avg.copy()
img_clean[img_std!=0] = 255
img_clean[img_clean==255] = 0
img_clean = minimum(img_clean.astype('uint8'), disk(3))


#%% Plot

fig, ax = plt.subplots() 
# plt.imshow(img, cmap='gray')
# plt.imshow(img_avg, cmap='gray')
# plt.imshow(img_std, cmap='gray')
plt.imshow(img_clean, cmap='gray')
ax.set_axis_off()

#%% Save

io.imsave(
    Path('data', img_name.replace('.bmp', '_clean.tif')),
    img_clean.astype('uint8'),
    check_contrast=False,
    )
