# 1) set negative shap values to 0 (lmk if anyone disagrees and we can diacuss)
# 2) reduce to 3x3 using sums or averages by region
# 3) normalize so that the sum of all cells is a constant number, probably 1
# import pickle

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
import pickle
from skimage.measure import block_reduce

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
    shap_map0 = block_reduce(shap_map_0[i], block_size=(75, 75), func=np.mean)
    # shap_map0 = resize(shap_map_0[i], (3, 3), order=1, preserve_range=True)

    # normalize and get rid of null and Nan values if there are any
    shap_map0 = np.nan_to_num(shap_map0)
    shap_map0 = shap_map0 / np.sum(shap_map0)

    # add it back to the array
    shap_map_0_3x3.append(shap_map0)

for i in range(len(shap_map_1)):
    # negative values to 0
    shap_map_1[i] = np.clip(shap_map_1[i], a_min=0, a_max=None)

    # reshape the saliency map into a 3x3 array, using pooling
    shap_map1 = block_reduce(shap_map_1[i], block_size=(75, 75), func=np.mean)
    # shap_map1 = resize(shap_map_1[i], (3, 3), order=1, preserve_range=True)

    # normalize and get rid of null and Nan values if there are any
    shap_map1 = np.nan_to_num(shap_map1)
    shap_map1 = shap_map1 / np.sum(shap_map1)

    # add it back to the array
    shap_map_1_3x3.append(shap_map1)

for i in range(len(occlusion_map_0)):
    # negative values to 0
    occlusion_map_0[i] = np.clip(occlusion_map_0[i], a_min=0, a_max=None)

    # reshape the saliency map into a 3x3 array, using pooling
    occlusion_map0 = block_reduce(occlusion_map_0[i], block_size=(75, 75), func=np.mean)
    # occlusion_map0 = resize(occlusion_map0[i], (3, 3), order=1, preserve_range=True)

    # normalize and get rid of null and Nan values if there are any
    occlusion_map0 = np.nan_to_num(occlusion_map0)
    occlusion_map0 = occlusion_map0 / np.sum(occlusion_map0)

    # add it back to the array
    occlusion_map_0_3x3.append(occlusion_map0)

for i in range(len(occlusion_map_1)):
    # negative values to 0
    occlusion_map_1[i] = np.clip(occlusion_map_1[i], a_min=0, a_max=None)

    # reshape the saliency map into a 3x3 array, using pooling
    occlusion_map1 = block_reduce(occlusion_map_1[i], block_size=(75, 75), func=np.mean)
    # occlusion_map1 = resize(occlusion_map_1[i], (3, 3), order=1, preserve_range=True)

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
with open('shap_and_occlusion_maps.pickle', 'wb') as f:
    pickle.dump(data, f)

# visualize the new map
# data1 = pd.read_pickle('shap_and_occlusion_maps_copy.pickle')

nr_imgs = len(data)
fig, axes = plt.subplots(nrows=nr_imgs, ncols=4, figsize=(40, nr_imgs * 4))
# fig.tight_layout()

axes[0, 0].set_title("SHAP\n(not action)")

axes[0, 1].set_title("SHAP 3x3\n(not action)")

axes[0, 2].set_title("SHAP\n(action)")

axes[0, 3].set_title("SHAP 3X3\n(action)")

# axes[0, 4].set_title("Occlusion\n(NOT action)")
#
# axes[0, 5].set_title("Occlusion 3x3\n(NOT action)")
#
# axes[0, 6].set_title("Occlusion\n(action)")
#
# axes[0, 7].set_title("Occlusion 3x3\n(action)")

# make a color map
from matplotlib.colors import LinearSegmentedColormap

colors = []
for l in np.linspace(1, 0, 100):
    colors.append((245 / 255, 39 / 255, 87 / 255, l))
for l in np.linspace(0, 1, 100):
    colors.append((24 / 255, 196 / 255, 93 / 255, l))

cm = LinearSegmentedColormap.from_list("shap", colors)

for index, row in data.iterrows():
    axes[index, 0].matshow(row['shap_0'], cmap=cm)
    axes[index, 0].axis('off')

    axes[index, 1].matshow(row['shap0_3x3'])
    axes[index, 1].axis('off')

    axes[index, 2].matshow(row['shap_1'], cmap=cm)
    axes[index, 2].axis('off')

    axes[index, 3].matshow(row['shap1_3x3'])
    axes[index, 3].axis('off')

    # axes[index, 4].matshow(row['occlusion_0'])
    # axes[index, 4].axis('off')
    #
    # axes[index, 5].matshow(row['occlusion0_3x3'])
    # axes[index, 5].axis('off')
    #
    # axes[index, 6].matshow(row['occlusion_1'])
    # axes[index, 6].axis('off')
    #
    # axes[index, 7].matshow(row['occlusion1_3x3'])
    # axes[index, 7].axis('off')

plt.show()
    # axes[2].matshow(heatmap1)
    # axes[2].set_title("Action")
    # axes[0].imshow(img_array[0])

    # plt.savefig('all_maps.pdf')

