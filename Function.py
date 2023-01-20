import pandas as pd
import os

path = "C:/Users/Nicki/Documents/GitHub/PR-Group-Project"
os.chdir(path)

df = pd.read_excel('Final responses spreadsheet.xlsx')
df = df.drop(df.columns[0], axis=1)

every_fourth_column = df.iloc[:, ::4]
columns = every_fourth_column.columns

for x in columns:
    # print(df[=".  Is this an action movie?"])
    print(df[x])

# df = df.assign(list_0_survey=[], list_1_survey=[])
