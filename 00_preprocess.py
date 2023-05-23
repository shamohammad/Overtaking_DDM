import pandas as pd
import glob
import shutil
import os
import numpy as np

# Read in the measures_vel.csv file
df = pd.read_csv("processed/measures/measures_vel.csv", sep=';')

#Filter data, remove unrealistic response times and missing values
T_dur = 4.0
T_min = 0.5
df = df[(df.Response_Time < T_dur) & (df.Response_Time > T_min)]
df = df[~df['Decision'].isin(['not finish overtake', 'calculation error', 'simulation error'])]
df = df[df['a_mean'].notnull()]

# Create the 'filtered_data' folder if it doesn't already exist
if not os.path.exists('filtered_data'):
    os.makedirs('filtered_data')

# cluster the v_start values
terciles = df["v_start"].quantile([0.33, 0.66])

def assign_tercile_label(x):
    if x <= terciles[0.33]:
        return "T1"
    elif x <= terciles[0.66]:
        return "T2"
    else:
        return "T3"

df["v_start_cluster"] = df["v_start"].apply(assign_tercile_label)
tercile_means = df.groupby("v_start_cluster")["v_start"].mean()
df["v_start_cluster"].replace({"T1": tercile_means[0], "T2": tercile_means[1], "T3": tercile_means[2]},
                                        inplace=True)

# Create a dictionary to store the file paths for the individual data files
file_dict = {}

for filename in glob.glob('processed/*.csv'):
    parts = filename.split('_')
    participant_id = parts[1]
    run = int(parts[3].split('.')[0])
    trial = int(parts[3].split('.')[1])
    file_dict[(participant_id, run, trial)] = filename

# Copy the relevant CSV files to the 'filtered_data' folder and add the new column
for index, row in df.iterrows():
    participant_id = row['Participant_ID']
    run = int(row['Run'])
    trial = int(row['Trial'])
    tta = int(np.floor(row['Condition_TTA']))
    dist = row['Condition_Dist']
    v_start_cluster = row['v_start_cluster']
    filename = f"processed/participant_{participant_id}_trial_{run}.{trial}_condition_tta_{tta}_condition_distance_{dist}.csv"
    # check if the file exists and add the new column if it does
    if (participant_id, run, trial) in file_dict:
        filepath = file_dict[(participant_id, run, trial)]
        individual_df = pd.read_csv(filepath)
        individual_df['v_start_cluster'] = v_start_cluster
        individual_df.to_csv(filepath, index=False, sep=";")

    try:
        shutil.copy(filename, 'filtered_data')
        print(f"Copied {filename} to filtered_data folder")
    except FileNotFoundError:
        print(f"File {filename} not found")


directory = "filtered_data/"

# Loop through all the files in the directory
for filename in os.listdir(directory):
    # Check if the file is a CSV file
    if filename.endswith(".csv"):
        # Open the file in read mode and read its contents as text
        with open(os.path.join(directory, filename), "r") as file:
            text = file.read()
        # Replace all commas with semicolons
        text = text.replace(",", ";")
        # Open the file again in write mode and write the modified text back to it
        with open(os.path.join(directory, filename), "w") as file:
            file.write(text)
