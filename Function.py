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
print(df2)
