import pandas as pd
import numpy as np
import os
import sys

os.chdir(os.path.dirname(sys.argv[0]))

df = pd.read_excel('Final responses spreadsheet.xlsx')

filtered_df1 = df.filter(regex='^Question ')
filtered_df2 = df.filter(regex='^[0-9]')

n_rows = df.shape[0]
timestamps = []
predictions = []
maps = []
ids =[940, 3073, 6545, 6912, 11360, 12015, 15783, 23413, 24283, 30364, 30625, 32357, 33433, 34789, 41850]
movie_ids = []

for x in range(0, n_rows):
    counter = 0
    id_counter = 0
    for y in filtered_df2.columns:
        a = np.zeros((3,3))
        timestamps.append(df["Timestamp"][x])
        pred = 1 if df[y][x] == "Yes, it is action" else 0
        predictions.append(pred)
        movie_ids.append(ids[id_counter])
        id_counter += 1

        for z in range(0, 3):
            if(type(df[filtered_df1.columns[counter]][x]) == float):
                elementList = []
            else:
                elementList = df[filtered_df1.columns[counter]][x].split()
                for l in elementList:
                    if l == "A," or l == 'A':
                        a[z,0] = 1
                    if l == "B," or l == 'B':
                        a[z,1] = 1
                    if l == "C":
                        a[z,2] = 1
            counter+= 1
        maps.append(a)
            
data = {"Timestamp": timestamps, "Movie ID": movie_ids, "Prediction": predictions, "Map": maps}
df2 = pd.DataFrame(data)

movie_ids2 = []
preds2 = []
maps2 = []

for id in ids:
    rows_pred_true = df2.loc[(df2["Movie ID"] == id) & (df2["Prediction"] == 1)]
    rows_pred_true2 = rows_pred_true["Map"]

    mean_array_true = np.mean(rows_pred_true2, axis=0)
    normalized_mean_array_true = mean_array_true / np.sum(mean_array_true)

    rows_pred_false = df2.loc[(df2["Movie ID"] == id) & (df2["Prediction"] == 0)]
    rows_pred_false2 = rows_pred_false["Map"]

    mean_array_false = np.mean(rows_pred_false2, axis=0)
    normalized_mean_array_false = mean_array_false / np.sum(mean_array_false)
    
    movie_ids2.extend([id, id])
    preds2.extend([1,0])
    maps2.extend([normalized_mean_array_true, normalized_mean_array_false])

data2 = {"Movie ID": movie_ids2, "Prediction": preds2, "Map": maps2}
df3 = pd.DataFrame(data2)

import matplotlib.pyplot as plt

# Create a figure with 10 rows and 3 columns of subplots
fig, axs = plt.subplots(5, 6, figsize=(30, 30))

# Generate 30 random 3x3 arrays
arrays = [np.random.rand(3, 3) for _ in range(30)]

print(arrays[0])
print(np.array(df3["Map"][0]))

# Plot each array in a separate subplot
for i, ax in enumerate(axs.flat):
    if i < 30:
        if np.isnan(df3["Map"][i]).any():
            ax.axis('off')
        else:
            ax.imshow(np.array(df3["Map"][i]))
            ax.axis('off')
        ax.set_title("ID: " + str(df3["Movie ID"][i]) + " Prediction: " + str(df3["Prediction"][i]), loc='center')
plt.subplots_adjust(wspace=0)
plt.show()


# df3.to_pickle('survey_answers.pkl')


# from scipy.stats import wasserstein_distance

# data_shap = pd.read_pickle('shap_and_occlusion_maps.pickle')

# print(data_shap["occlusion_0"][1])

# emd_human_shap = []
# emd_human_occlusion = []

# for id in ids:


# # define the two probability distributions
# distribution1 = [0.1, 0.3, 0.3, 0.2, 0.1]
# distribution2 = [0.2, 0.2, 0.2, 0.2, 0.2]

# calculate the EMD/Wasserstein distance
# emd = wasserstein_distance(distribution1, distribution2)
# print(emd)
