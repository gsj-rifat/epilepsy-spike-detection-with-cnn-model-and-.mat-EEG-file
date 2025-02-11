#I have 22 files of data, each file is used to plot the EEG signal in a loop

import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne import pick_channels
from scipy.io import loadmat

# Set the path to the folder containing the files
mat_path = 'C:\\masterthesis-golam-rifat\\pythonProject1\\Clean data\\Pt1'
txt_path = 'C:\\masterthesis-golam-rifat\\pythonProject1\\Clean data\\Pt1\\Events file sampled\\data_marked_by_ep_3'

# Get a list of all the files in the folder
mat_list = os.listdir(mat_path)
txt_list = os.listdir(
    'C:\\masterthesis-golam-rifat\\pythonProject1\\Clean data\\Pt1\\Events file sampled\\data_marked_by_ep_3')
def natural_sort_key(filename):
    # Extract the numeric part from the filename
    numeric_part = ''.join(c for c in filename if c.isdigit())
    return int(numeric_part) if numeric_part else filename

txt_list = sorted(txt_list, key=natural_sort_key)

# Loop through the list of files and read the .txt and .mat files
for file_name_1 in txt_list:
    # Check if the file has the .txt extension
    events = mne.read_events(os.path.join(txt_path, file_name_1))  # Load your events from a text file
    print(file_name_1)
    for file_name in mat_list:
        # Open the file and read its contents
        print(file_name)
        mat_data = loadmat(os.path.join(mat_path, file_name))
        data = np.transpose(np.array(mat_data['F']))
        #build metadata
        ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
                    'Fz',
                    'Cz', 'Pz']  # Replace with your actual channel names
        sfreq = 200  # Replace with your actual sampling rate
        info = mne.create_info(ch_names, sfreq)
        raw = mne.io.RawArray(data, info)
        n_times = data.shape[1]  # Number of time points
        times = np.arange(n_times) / sfreq  # Time array in seconds
        raw._set_times(times)
        montage = mne.channels.make_standard_montage('standard_1005')
        raw.set_montage(montage)
        stim_data = [0] * raw.n_times  # Create a new channel for the stimulus data
        stim_info = mne.create_info(['STI 014'], raw.info['sfreq'], ['stim'])
        stim_raw = mne.io.RawArray([stim_data], stim_info)

        # Add the stimulus channel to the Raw object
        raw.add_channels([stim_raw], force_update_info=True)
        raw.add_events(events, stim_channel='STI 014')
        raw_1= raw.copy()
        pick = raw_1.pick_channels([ 'F3', 'STI 014'])
        # Exclude the stimulus channel
        exclude = ['STI 014']
        exclude_stim = [ch for ch in raw.ch_names if ch not in exclude]

        raw.filter(l_freq=0.5, h_freq=60, picks=exclude_stim)

        raw.notch_filter(freqs=50, picks=exclude_stim)
        picks = raw.pick_channels([ 'F3', 'STI 014'])
        pick.plot(events)
        #plt.show()
        picks.plot(events)
        plt.show()

        # Save the plot to a file
        # fig.savefig('C:\masterthesis-golam-rifat\pythonProject1\Clean data\Pt1\Events file sampled\Plots\plots.jpg')
        break
