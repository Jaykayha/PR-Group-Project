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
maps2_0 = []
maps2_1 = []

for id in ids:
    rows_pred_true = df2.loc[(df2["Movie ID"] == id) & (df2["Prediction"] == 1)]
    rows_pred_true2 = rows_pred_true["Map"]

    mean_array_true = np.mean(rows_pred_true2, axis=0)
    normalized_mean_array_true = mean_array_true / np.sum(mean_array_true)

    rows_pred_false = df2.loc[(df2["Movie ID"] == id) & (df2["Prediction"] == 0)]
    rows_pred_false2 = rows_pred_false["Map"]

    mean_array_false = np.mean(rows_pred_false2, axis=0)
    normalized_mean_array_false = mean_array_false / np.sum(mean_array_false)
    
    movie_ids2.append(id)
    maps2_1.append(normalized_mean_array_true)
    maps2_0.append(normalized_mean_array_false)

data2 = {"Movie ID": movie_ids2, "Map_0": maps2_0, "Map_1": maps2_1}
df3 = pd.DataFrame(data2)

from scipy.stats import wasserstein_distance

data_shap = pd.read_pickle('shap_and_occlusion_maps.pickle')


emd_human_shap_0 = []
emd_human_shap_1 = []
emd_human_occlusion_0 = []
emd_human_occlusion_1 = []

data_shap = data_shap.drop(3)
data_shap = data_shap.reset_index(drop=True)

data_shap['id'] = data_shap['id'].astype(int)
data_shap = data_shap.sort_values(by='id', ascending=True)

# Occlusion 0
for i in range(0, 15):
    distribution1 = data_shap["occlusion0_3x3"][i].flatten()
    distribution2 = df3["Map_0"][i].flatten()
    emd = wasserstein_distance(distribution1, distribution2)

    emd_human_occlusion_0.append(emd)

# Occlusion 1
for i in range(0, 15):
    distribution1 = data_shap["occlusion1_3x3"][i].flatten()
    distribution2 = df3["Map_1"][i].flatten()
    emd = wasserstein_distance(distribution1, distribution2)


    emd_human_occlusion_1.append(emd)

# Shap 1
for i in range(0, 15):
    distribution1 = data_shap["shap0_3x3"][i].flatten()
    distribution2 = df3["Map_0"][i].flatten()
    emd = wasserstein_distance(distribution1, distribution2)


    emd_human_shap_0.append(emd)

# Shap 1
for i in range(0, 15):
    distribution1 = data_shap["shap1_3x3"][i].flatten()
    distribution2 = df3["Map_1"][i].flatten()
    emd = wasserstein_distance(distribution1, distribution2)


    emd_human_shap_1.append(emd)


data3 = {"id": ids,"Shap_to_Human_0": emd_human_shap_0, "Shap_to_Human_1": emd_human_shap_1, "Occlusion_to_Human_0": emd_human_occlusion_0, "Occlusion_to_Human_1": emd_human_occlusion_1}
df4 = pd.DataFrame(data3)

mean1 = np.nanmean(emd_human_shap_0)
mean2 = np.nanmean(emd_human_shap_1)
mean3 = np.nanmean(emd_human_occlusion_0)
mean4 = np.nanmean(emd_human_occlusion_1)

print("Human shap 0: " + str(mean1))
print("Human shap 1: " + str(mean2))
print("Human occlusion 0: " + str(mean3))
print("Human occlusion 1: " + str(mean4))
# df4.to_csv('out.csv')  

# for id in ids:

# # define the two probability distributions
# distribution1 = [0.1, 0.3, 0.3, 0.2, 0.1]
# distribution2 = [0.2, 0.2, 0.2, 0.2, 0.2]

# calculate the EMD/Wasserstein distance
# emd = wasserstein_distance(distribution1, distribution2)
# print(emd)


# import matplotlib.pyplot as plt

# # Create a figure with 10 rows and 3 columns of subplots
# fig, axs = plt.subplots(5, 6, figsize=(30, 30))

# # Generate 30 random 3x3 arrays
# arrays = [np.random.rand(3, 3) for _ in range(30)]

# print(arrays[0])
# print(np.array(df3["Map"][0]))

# # Plot each array in a separate subplot
# for i, ax in enumerate(axs.flat):
#     if i < 30:
#         if np.isnan(df3["Map"][i]).any():
#             ax.axis('off')
#         else:
#             ax.imshow(np.array(df3["Map"][i]))
#             ax.axis('off')
#         ax.set_title("ID: " + str(df3["Movie ID"][i]) + " Prediction: " + str(df3["Prediction"][i]), loc='center')
# plt.subplots_adjust(wspace=0)
# plt.show()


# # df3.to_pickle('survey_answers.pkl')


