import pandas as pd
import os
import pytz
from datetime import datetime

# Specify the directory containing your CSV files
directory_path = 'datasets/demodag_5-3_raw'

# dont know what the second column means
column_names = ['PN', 'something', 'UNIX Timestamp', 'Accelerometer Data']

dataframes_list = []

for file_name in os.listdir(directory_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(directory_path, file_name)
        df = pd.read_csv(file_path, names=column_names, header=None)
        dataframes_list.append(df)

# combine all  DataFrames in the list into a single DataFrame
combined_df = pd.concat(dataframes_list, ignore_index=True)

sorted_df = combined_df.sort_values(by='UNIX Timestamp')

sorted_csv_path = 'datasets/demodag1_unlabeled.csv'
sorted_df.to_csv(sorted_csv_path, index=False)


labels_df = pd.read_csv('datasets/demodag1_annotations.csv')

date = '2024-03-05'
# Add one hour for GMT time differnce and convert to millisecond format like the tag has
labels_df['From'] = (pd.to_datetime(date + ' ' + labels_df['From']).astype(int) / 10**9 * 1000) - 3600000
labels_df['To'] = (pd.to_datetime(date + ' ' + labels_df['To']).astype(int) / 10**9 * 1000) - 3600000

sorted_df['Behavior'] = None

# Loop through the labels DataFrame and assign labels to the accelerometer data
for index, row in labels_df.iterrows():
    mask = (sorted_df['UNIX Timestamp'] >= row['From']) & (sorted_df['UNIX Timestamp'] < row['To'])
    sorted_df.loc[mask, 'Behavior'] = row['Behavior']



# Save the labeled DataFrame back to CSV
labeled_csv_path = 'datasets/demodag1.csv'
sorted_df.to_csv(labeled_csv_path, index=False)


