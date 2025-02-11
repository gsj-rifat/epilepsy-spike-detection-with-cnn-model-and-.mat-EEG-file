import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy import signal
import os

mne.set_log_level('error')  # reduce extraneous MNE output

# data = pd.read_csv('C:\\Thesis\\data02_ch4.csv')
path = 'C:\\masterthesis-golam-rifat\\pythonProject1\\Clean data\\Pt1\\'
name = 'data022_F_ch.mat'
data = loadmat(path + name)
data = np.array(data['F'])
data = np.transpose(data)
# Extract channel names

ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz',
            'Cz', 'Pz']

# Replace with your actual channel names , 'Fp2' ,'Cz', 'Pz'
# Define the sampling frequency
sfreq = 200  # Replace with your actual sampling rate
info = mne.create_info(ch_names, sfreq, ch_types='eeg')

raw = mne.io.RawArray(data, info);  # Add time information to the Raw object

# Set the montage
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage)

n_times = data.shape[1]  # Number of time points
times = np.arange(n_times) / sfreq  # Time array in seconds

# Add time information to your DataFrame
raw._set_times(times)

# Apply a 50 Hz notch filter
raw.notch_filter(freqs=50, picks=ch_names)
# raw.plot()
# plt.show()
# Apply a band-pass filter between 1 and 50 Hz
raw.filter(l_freq=0.5, h_freq=60, picks=ch_names)
# Convert data from μV to mV

data1 = raw.get_data()
fs = 200  # Sampling frequency
epoch_length = 1  # Epoch length in seconds
nperseg = 50 #epoch_length * fs  # Number of samples per conent
noverlap = 25 #nperseg / 2   # Overlap between segments
i=0


for i in range(19):
    f, t, Sxx = signal.spectrogram(data1[i], fs, window='blackman', nperseg=nperseg, noverlap=noverlap,
                               nfft=256)
    df = pd.DataFrame(Sxx, columns=t, index=f)
    file_prefix = 'data_'
    folder_prefix = 'Folder_'
    folder_name = 'C:\\masterthesis-golam-rifat\\Feature folder\\Spectro Feature\\' + folder_prefix + name
    os.makedirs(folder_name, exist_ok=True)
    channel= raw.ch_names[i]
    # Save the DataFrame to CSV
    csv_file_path = os.path.join(folder_name, file_prefix + channel + '.csv')
    df.to_csv(csv_file_path, index_label='Freq')

print(t)

# Save to CSV
#df.to_csv('spectrogram_data.csv', index_label='Freq')