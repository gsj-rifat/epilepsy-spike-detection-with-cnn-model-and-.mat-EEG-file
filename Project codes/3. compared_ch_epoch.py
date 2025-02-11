#In this code, the nearby channels are substracted so that the spikes are more visible and only epochs are created

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

mne.set_log_level('error')  # reduce extraneous MNE output

data = loadmat('C:\\masterthesis-golam-rifat\\pythonProject1\\Clean data\\Pt1\\data001_F_ch.mat')
data = np.array(data['F'])
data = np.transpose(data)

# Define channel names
ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz',
            'Cz', 'Pz']  # Replace with your actual channel names , 'Fp2' ,'Cz', 'Pz'
sfreq = 200  # Replace with your actual sampling rate
info = mne.create_info(ch_names, sfreq, ch_types='eeg')
raw = mne.io.RawArray(data, info);  # Add time information to the Raw object

# Set the montage
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage)

# Set the reference
#raw, ref_data = mne.set_eeg_reference(raw, ref_channels=['Cz', 'Fz'])

n_times = data.shape[1]  # Number of time points
times = np.arange(n_times) / sfreq  # Time array in seconds

# Add time information to your DataFrame
raw._set_times(times)

# Build the stimulus channel
stim_data = [0] * raw.n_times  # Create a new channel for the stimulus data
stim_info = mne.create_info(['STI 014'], raw.info['sfreq'], ['stim'])
stim_raw = mne.io.RawArray([stim_data], stim_info)

fig = mne.viz.plot_sensors(raw.info, show_names=True)
fig.suptitle(f"EEG Electrode Positions ({montage})")
plt.show()

# Add the stimulus channel to the Raw object
raw.add_channels([stim_raw], force_update_info=True)

# Load your events from a text file
events = mne.read_events(
    'C:\\masterthesis-golam-rifat\\pythonProject1\\Clean data\\Pt1\\Events file sampled\\data_marked_by_ep_1\\Pt1_r1_data_marked_by_ep_1-eve.txt')

# Add the events to the Raw object
raw.add_events(events, stim_channel='STI 014')
print(raw.info)
raw.plot(events)
plt.plot()
mne.viz.plot_events(events, sfreq=raw.info['sfreq'])
exclude = ['STI 014']
exclude_stim = [ch for ch in raw.ch_names if ch not in exclude]
# Apply a 50 Hz notch filter
raw.notch_filter(freqs=50, picks=exclude_stim)
# Apply a band-pass filter between 1 and 50 Hz
raw.filter(l_freq=1, h_freq=60, picks=exclude_stim)

raw.plot(events)
plt.show()

raw.drop_channels(['T3', 'T4', 'T5', 'T6'])
anode = ['Fz', 'Cz', 'Fp1', 'F3', 'C3', 'P3', 'Fp2', 'F4', 'C4', 'P4', 'Fp1', 'Fp2']
cathode = ['Cz', 'Pz', 'F3', 'C3', 'P3', 'O1', 'F4', 'C4', 'P4', 'O2', 'F7', 'F8']
raw1 = mne.set_bipolar_reference(raw, anode=anode, cathode=cathode)

raw1.apply_function(lambda x: x - x.mean())
raw1.pick_channels(['F3-C3'])
raw1.plot(events, event_color= 'r')
plt.show()

#raw1.pick_channels(['Fp1-F3', 'F3-C3'])
#creating separate epochs
epochs = mne.Epochs(raw1, events, event_id=None, tmin=-0.5, tmax=0.5, baseline=(None, 0), picks=None, preload=False,
                    reject=None, flat=None, proj=True, decim=1, reject_tmin=None, reject_tmax=None, detrend=None,
                    on_missing='raise', reject_by_annotation=True, metadata=None, event_repeated='error', verbose=None)

#creating overlapping epochs
epochs1 = mne.make_fixed_length_epochs(raw1, duration=1.0, overlap=0.2)
mne.Epochs.plot(epochs1, scalings=100)
plt.show()
print(len(epochs1))