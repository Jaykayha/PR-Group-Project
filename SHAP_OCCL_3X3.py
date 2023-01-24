# 1) set negative shap values to 0 (lmk if anyone disagrees and we can diacuss)
# 2) reduce to 3x3 using sums or averages by region
# 3) normalize so that the sum of all cells is a constant number, probably 1
import pickle

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
from skimage.transform import resize

# load the pickle data
data = pd.read_pickle('shap_and_occlusion_maps.pickle')

# extract SHAP and Occlusion maps
shap_map_0 = data.shap_0
shap_map_1 = data.shap_1

occlusion_map_0 = data.occlusion_0
occlusion_map_1 = data.occlusion_1

# initialize reshaped and normalized maps
shap_map_0_3x3 = []
shap_map_1_3x3 = []

occlusion_map_0_3x3 = []
occlusion_map_1_3x3 = []

for i in range(len(shap_map_0)):
    # negative values to 0
    shap_map_0[i] = np.clip(shap_map_0[i], a_min=0, a_max=None)

    # reshape the saliency map into a 3x3 array, using pooling
    shap_map0 = block_reduce(shap_map_0[i], block_size=(3, 3), func=np.mean)
    shap_map0 = resize(shap_map_0[i], (3, 3), order=1, preserve_range=True)

    # normalize and get rid of null and Nan values if there are any
    shap_map0 = np.nan_to_num(shap_map0)
    shap_map0 = shap_map0 / np.sum(shap_map0)

    # add it back to the array
    shap_map_0_3x3.append(shap_map0)

for i in range(len(shap_map_1)):
    # negative values to 0
    shap_map_1[i] = np.clip(shap_map_1[i], a_min=0, a_max=None)

    # reshape the saliency map into a 3x3 array, using pooling
    shap_map1 = block_reduce(shap_map_1[i], block_size=(3, 3), func=np.mean)
    shap_map1 = resize(shap_map_1[i], (3, 3), order=1, preserve_range=True)

    # normalize and get rid of null and Nan values if there are any
    shap_map1 = np.nan_to_num(shap_map1)
    shap_map1 = shap_map1 / np.sum(shap_map1)

    # add it back to the array
    shap_map_1_3x3.append(shap_map1)

for i in range(len(occlusion_map_0)):
    # negative values to 0
    occlusion_map_0[i] = np.clip(occlusion_map_0[i], a_min=0, a_max=None)

    # reshape the saliency map into a 3x3 array, using pooling
    occlusion_map0 = block_reduce(occlusion_map_0[i], block_size=(3, 3), func=np.mean)
    occlusion_map0 = resize(occlusion_map0[i], (3, 3), order=1, preserve_range=True)

    # normalize and get rid of null and Nan values if there are any
    occlusion_map0 = np.nan_to_num(occlusion_map0)
    occlusion_map0 = occlusion_map0 / np.sum(occlusion_map0)

    # add it back to the array
    occlusion_map_0_3x3.append(occlusion_map0)

for i in range(len(occlusion_map_1)):
    # negative values to 0
    occlusion_map_1[i] = np.clip(occlusion_map_1[i], a_min=0, a_max=None)

    # reshape the saliency map into a 3x3 array, using pooling
    occlusion_map1 = block_reduce(occlusion_map_1[i], block_size=(3, 3), func=np.mean)
    occlusion_map1 = resize(occlusion_map_1[i], (3, 3), order=1, preserve_range=True)

    # normalize and get rid of null and Nan values if there are any
    occlusion_map1 = np.nan_to_num(occlusion_map1)
    occlusion_map1 = occlusion_map1 / np.sum(occlusion_map1)

    # add it back to the array
    occlusion_map_1_3x3.append(occlusion_map1)

#add the 3x3 maps as new columns in data
data['shap0_3x3'] = shap_map_0_3x3
data['shap1_3x3'] = shap_map_1_3x3
data['occlusion0_3x3'] = occlusion_map_0_3x3
data['occlusion1_3x3'] = occlusion_map_1_3x3

#save the data back into pickle file
#TODO save to pickle file
#with open('shap_and_occlusion_maps.pickle', 'wb') as f:
#    pickle.dump(data, f)

# visualize the new maps
#TODO Fix diplaying of maps
for i in range(len(data.shap_0)):
    plt.figure()
    plt.imshow(shap_map_0[i])
    plt.title('Original SHAP')

    plt.figure()
    plt.imshow(shap_map_0_3x3[i])
    plt.title('3x3 SHAP')

