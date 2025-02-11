#in this code, the features are generated like the previous one and saved in a csv file. But there are three sets of synthetic data along with 
# the main data. events = mne.make_fixed_length_events(raw, start=0.75, stop=None, duration=1.0) in this line, 0.75 is one kind of data generating. We can rplace the 0.75 with 0.50 and 0.25 for other 2 sets of data. This will generate 4 sets of feature.

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import hilbert
from scipy.stats import skew, kurtosis, entropy
import os

mne.set_log_level('error')  # reduce extraneous MNE output

# data = pd.read_csv('C:\\Thesis\\data02_ch4.csv')
path = 'C:\\Thesis\\data\\Pt1\\Pt1\\'
name = 'data022_F_ch.mat'
data = loadmat(path + name)
data = np.array(data['F'])
data = np.transpose(data)
# Extract channel names

ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz',
            'Cz', 'Pz']

# Replace with your actual channel names , 'Fp2' ,'Cz', 'Pz'
sfreq = 200  # Replace with your actual sampling rate
info = mne.create_info(ch_names, sfreq, ch_types='eeg')
# Define the sampling frequency
raw = mne.io.RawArray(data, info);  # Add time information to the Raw object

# Set the montage
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage)

n_times = data.shape[1]  # Number of time points
times = np.arange(n_times) / sfreq  # Time array in seconds

# Add time information to your DataFrame
raw._set_times(times)

raw.notch_filter(freqs=50, picks=ch_names)

# Apply a band-pass filter between 1 and 50 Hz
raw.filter(l_freq=1, h_freq=60, picks=ch_names)


events = mne.make_fixed_length_events(raw, start=0.75, stop=None, duration=1.0)
epochs1 = mne.Epochs(raw, events, tmin=0., tmax=1., baseline=None, preload=True)



for channels in raw.ch_names:
    epochs_mav = epochs1.copy().pick_channels([channels])
    mav_values = epochs_mav.get_data().mean(axis=2)  # Compute mean across time points
    std_values = epochs_mav.get_data().std(axis=2)  # Compute standard deviation
    skewness_values = skew(epochs_mav.get_data(), axis=2)  # skewness of amplitude values
    kurtosis_values = kurtosis(epochs_mav.get_data(), axis=2)
    ptp_values = np.abs(epochs_mav.get_data()).ptp(axis=2)  # Absolute PTP
    # freqs, psd = mne.time_frequency.psd_array_multitaper(epochs_mav.get_data(), fmin=0.5, fmax=50, n_jobs=1, sfreq=sfreq)

    # fig = mne.viz.plot_epochs_psd(epochs, fmin=.5, fmax=50, average=True)

    # Compute features for each epoch
    amplitude_features = []
    frequency_features = []
    phase_features = []
    kfd_values = []
    entropy_values = []

    for epoch in epochs_mav:
        data = epoch[0]  # Assuming single-channel EEG
        # Compute amplitude (root mean square)
        amplitude = np.sqrt(np.mean(data ** 2))
        amplitude_features.append(amplitude)
        # Compute frequency using Hilbert transform
        analytic_signal = hilbert(data)
        instantaneous_phase = np.angle(analytic_signal)
        frequency = np.diff(instantaneous_phase) / (2 * np.pi) * 200
        frequency_features.append(frequency.mean())
        # Compute phase (mean phase angle)
        phase = np.mean(instantaneous_phase)
        phase_features.append(phase)
        total_length = np.sum(np.abs(np.diff(data)))  # Compute total length
        # Compute KFD
        N = len(data)
        L0 = np.abs(data[-1] - data[0])
        kfd = np.log(N) / np.log(total_length / L0)
        kfd_values.append(kfd)
        prob_distribution = np.abs(data) / np.sum(np.abs(data))  # Normalize to probabilities
        shannon_entropy = entropy(prob_distribution, base=2)  # Base 2 for bits
        entropy_values.append(shannon_entropy)

    feature_data = {
        'Epoch': range(1, len(mav_values) + 1),
        'MAV': mav_values.flatten(),
        'Standard_Deviation': std_values.flatten(),
        'Skewness': skewness_values.flatten(),
        'Kurtosis': kurtosis_values.flatten(),
        'Peak to peak': ptp_values.flatten(),
        'Amplitude': amplitude_features,
        'Frequency': frequency_features,
        'Phase': phase_features,
        'Katz fractal dimension': kfd_values,
        'Shannons Entropy': entropy_values
    }

    # Create a DataFrame
    df = pd.DataFrame(feature_data)
    file_prefix = 'data_'
    folder_prefix = 'folder_'
    folder_name = 'C:\\Thesis\\Feature folder\\' + folder_prefix + name
    os.makedirs(folder_name, exist_ok=True)
    # np.savetxt(file_name, df, delimiter=',')
    # Save the DataFrame to CSV
    csv_file_path = os.path.join(folder_name, file_prefix + channels + '.csv')
    df.to_csv(csv_file_path, index=False)

    print(f"Features saved to {csv_file_path}")
