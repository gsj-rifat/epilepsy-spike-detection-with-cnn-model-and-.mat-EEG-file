#in this code, the time points of the spikes (which are labels) are shifted as per the synthetic data. So .25, .5, .75 is added.

import os
import pandas as pd

# Specify the source directory and target directory
source_dir = 'C:\masterthesis-golam-rifat\pythonProject1\Clean data\Pt1\Event files time points\data_marked_by_ep_3'
target_dir = 'C:\masterthesis-golam-rifat\Feature folder\Spike labels\Event files time points shift.75\data_marked_by_ep_3'

# Create the target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Get a list of all text files in the source directory
text_files = [f for f in os.listdir(source_dir) if f.endswith('.txt')]

for file_name in text_files:
    # Construct the full file path
    file_path = os.path.join(source_dir, file_name)

    # Read the text file into a pandas DataFrame
    df = pd.read_csv(file_path, header=None, delim_whitespace=True)

    # Add 0.5 to all numbers in the DataFrame
    df = df - 0.75

    # Construct the full target file path
    target_file_path = os.path.join(target_dir, file_name)

    # Write the modified DataFrame back to a text file
    df.to_csv(target_file_path, header=False, index=False, sep=' ')
