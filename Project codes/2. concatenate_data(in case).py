#if you have more than one data file, then this code is to concatenate the all the data with metadata


import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne import concatenate_raws
from scipy.io import loadmat

# marge data, embade events and plot with mne

data2 = loadmat('C:\\masterthesis-golam-rifat\\pythonProject1\\Clean data\Pt2\\EEG_ch4\\data02_ch4.mat')
data3 = loadmat('C:\\masterthesis-golam-rifat\\pythonProject1\\Clean data\Pt2\\EEG_ch4\\data03_ch4.mat')
data4 = loadmat('C:\\masterthesis-golam-rifat\\pythonProject1\\Clean data\Pt2\\EEG_ch4\\data04_ch4.mat')
data5 = loadmat('C:\\masterthesis-golam-rifat\\pythonProject1\\Clean data\Pt2\\EEG_ch4\\data05_ch4.mat')
data6 = loadmat('C:\\masterthesis-golam-rifat\\pythonProject1\\Clean data\Pt2\\EEG_ch4\\data06_ch4.mat')

data2 = np.array(data2['ans'])
data3 = np.array(data3['ans'])
data4 = np.array(data4['ans'])
data5 = np.array(data5['ans'])
data6 = np.array(data6['ans'])

#build metadata

ch_names = ['Fp1', 'Fp2', 'Cz', 'Pz']  # Replace with your actual channel names
sfreq = 2400  # Replace with your actual sampling rate
info = mne.create_info(ch_names, sfreq, ch_types='eeg')
raw2 = mne.io.RawArray(data2, info)
raw3 = mne.io.RawArray(data3, info)
raw4 = mne.io.RawArray(data4, info)
raw5 = mne.io.RawArray(data5, info)
raw6 = mne.io.RawArray(data6, info)

n_times2 = data2.shape[1]  # Number of time points
n_times3 = data3.shape[1]
n_times4 = data4.shape[1]
n_times5 = data5.shape[1]
n_times6 = data6.shape[1]

times2 = np.arange(n_times2) / sfreq  # Time array in seconds
times3 = np.arange(n_times3) / sfreq
times4 = np.arange(n_times4) / sfreq
times5 = np.arange(n_times5) / sfreq
times6 = np.arange(n_times6) / sfreq

raw2._set_times(times2)  #setting times for all raw data
raw3._set_times(times3)
raw4._set_times(times4)
raw5._set_times(times5)
raw6._set_times(times6)

# Concatenate the raw data
raw = concatenate_raws([raw6, raw5, raw4, raw3, raw2])
# Remove boundary annotations
raw.annotations.delete(raw.annotations.description == 'BAD boundary')

# raw.plot(scalings=dict(eeg=1e-6))
# plt.show()
stim_data = [0] * raw.n_times  # Create a new channel for the stimulus data
stim_info = mne.create_info(['STI 014'], raw.info['sfreq'], ['stim'])
stim_raw = mne.io.RawArray([stim_data], stim_info)

# Add the stimulus channel to the Raw object
raw.add_channels([stim_raw], force_update_info=True)

# Load your events from a text file
events = mne.read_events('C:\\masterthesis-golam-rifat\\pythonProject1\\markingfile-eve.txt')
# events = [[193294, 0, 45]]
# Add the events to the Raw object
raw.add_events(events, stim_channel='STI 014')
# print(raw.info)
# mne.viz.plot_events(events, sfreq=raw.info['sfreq'])
raw.plot()
plt.show()
# Apply a 50 Hz notch filter
raw.notch_filter(freqs=50, picks=raw.ch_names)
raw.plot()
plt.show()
# Apply a band-pass filter between 1 and 50 Hz
raw.filter(l_freq=1, h_freq=50, picks=raw.ch_names)
raw.plot()
plt.show()
