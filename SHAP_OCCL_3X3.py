# 1) set negative shap values to 0 (lmk if anyone disagrees and we can diacuss)
# 2) reduce to 3x3 using sums or averages by region
# 3) normalize so that the sum of all cells is a constant number, probably 1

# Also lets add these new processed maps all as new columns in the same dataframe
# we already created and imported from drive.
# This will make the info easier to manage


# Resize and normalize SHAP to 3x3
# values in the 3x3 grid will not be the average of the original data but INTERPOLATED values.
# shap_map_0_resized = resize(shap_map_0, (3, 3), mode='reflect', anti_aliasing=True)
# shap_map_1_resized = resize(shap_map_1, (3, 3), mode='reflect', anti_aliasing=True)
#
# occlusion_map_0_resized = resize(occlusion_map_0, (3, 3), mode='reflect', anti_aliasing=True)
# occlusion_map_1_resized = resize(occlusion_map_1, (3, 3), mode='reflect', anti_aliasing=True)


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.measure import block_reduce
from scipy.signal import convolve2d
from skimage.transform import resize

# load the pickle data
data = pd.read_pickle('shap_and_occlusion_maps.pickle')
print(data)

# extract SHAP and Occlusion maps
shap_map_0 = data.shap_0
print(data.filename[7])
occlusion_map_0 = data.occlusion_0

shap_map_1 = data.shap_1
occlusion_map_1 = data.occlusion_1

# initialize reshaped and normalized maps
shap_map_0_3x3 = []
shap_map_1_3x3 = []

occlusion_map_0_3x3 = []
occlusion_map_1_3x3 = []

for i in range(len(shap_map_0)):
    # negative values to 0
    shap_map_0[i] = np.clip(shap_map_0[i], a_min=0, a_max=None)

    # reshape the saliency map into a 3x3 array
    #shap_map0 = np.reshape(shap_map_0[i], (3, 3))
    # filter
    #avg_filter = np.ones((3, 3)) / 9
    #shap_map0 = convolve2d(shap_map_0[i], avg_filter, mode='valid')
    # take the average across each 3x3 cell
    #shap_map0 = np.mean(shap_map0, axis=(0, 1))

    shap_map0 = block_reduce(shap_map_0[i], block_size=(3, 3), func=np.mean)
    # average the values in each cell
    shap_map0 = resize(shap_map_0[i], (3, 3), order=1, preserve_range=True)
    # shap_map0 = np.mean(shap_map_0[i], axis=(1))
    # shap_map0 = shap_map0.reshape(3,3)

    # normalize and get rid of null and Nan values if there are any
    shap_map0 = np.nan_to_num(shap_map0)
    shap_map0 = shap_map0 / np.sum(shap_map0)
    # add it back to the array
    print(shap_map0.shape)
    shap_map_0_3x3.append(shap_map0)

# visualize the grid
plt.imshow(shap_map_0_3x3[1])
plt.colorbar()
plt.show()
