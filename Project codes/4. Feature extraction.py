#In this code features are extracted from the substracted channels. But it can be done from the single channels also.

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import hilbert
from scipy.stats import skew, kurtosis, entropy

mne.set_log_level('error')  # reduce extraneous MNE output

# data = pd.read_csv('C:\\Thesis\\data02_ch4.csv')
data = loadmat('C:\\Thesis\\data\\Pt1\\Pt1\\data001_F_ch.mat')
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

# Apply a 50 Hz notch filter
raw.notch_filter(freqs=50, picks=ch_names)

# Apply a band-pass filter between 1 and 50 Hz
raw.filter(l_freq=1, h_freq=60, picks=ch_names)

raw.drop_channels(['T3', 'T4', 'T5', 'T6'])
anode = ['Fz', 'Cz', 'Fp1', 'F3', 'C3', 'P3', 'Fp2', 'F4', 'C4', 'P4', 'Fp1', 'Fp2']
cathode = ['Cz', 'Pz', 'F3', 'C3', 'P3', 'O1', 'F4', 'C4', 'P4', 'O2', 'F7', 'F8']
raw1 = mne.set_bipolar_reference(raw, anode=anode, cathode=cathode)

# raw1.pick_channels(['Fp1-F3', 'F3-C3'])

epochs = mne.Epochs(raw1, events, event_id=None, tmin=-0.5, tmax=0.5, baseline=(None, 0), picks=None, preload=True,
                    reject=None, flat=None, proj=True, decim=1, reject_tmin=None, reject_tmax=None, detrend=None,
                    on_missing='raise', reject_by_annotation=True, metadata=None, event_repeated='error', verbose=None)

# creating overlapping epochs
epochs1 = mne.make_fixed_length_epochs(raw1, duration=1.0, overlap=0.2)
# mne.Epochs.plot(epochs1, scalings=100,picks=['Fp1-F3', 'F3-C3'])
# mne.Epochs.plot(epochs, scalings=10,picks=['Fp1-F3', 'F3-C3'])
# plt.show()
for channels in raw1.ch_names
epochs_mav = epochs.copy().pick_channels(['F3-C3'])
mav_values = epochs_mav.get_data().mean(axis=2)  # Compute mean across time points
std_values = epochs_mav.get_data().std(axis=2)  # Compute standard deviation
skewness_values = skew(epochs_mav.get_data(), axis=2) # skewness of amplitude values
kurtosis_values = kurtosis(epochs_mav.get_data(), axis=2)
ptp_values = np.abs(epochs_mav.get_data()).ptp(axis=2)  # Absolute PTP
#freqs, psd = mne.time_frequency.psd_array_multitaper(epochs_mav.get_data(), fmin=0.5, fmax=50, n_jobs=1, sfreq=sfreq)

#fig = mne.viz.plot_epochs_psd(epochs, fmin=.5, fmax=50, average=True)


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
    'Amplitude' : amplitude_features,
    'Frequency': frequency_features,
    'Phase': phase_features,
    'Katz fractal dimension': kfd_values,
    'Shannons Entropy': entropy_values
}

# Create a DataFrame
#df = pd.DataFrame(feature_data)

# Specify the path where you want to save the CSV file
#csv_file_path = 'C:\masterthesis-golam-rifat\Feature folder\F3-C3_erp_features.csv'

# Save the DataFrame to CSV
df.to_csv(csv_file_path, index=False)

print(f"Features saved to {csv_file_path}")


#'visualizing features'
#Amplitude Across Epochs:

plt.figure(figsize=(10, 6))
plt.plot(range(len(amplitude_features)), amplitude_features, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Amplitude')
plt.title('Amplitude Across Epochs')
plt.grid(True)
plt.show()

#Frequency Across Epochs:
plt.figure(figsize=(10, 6))
plt.plot(range(len(frequency_features)), frequency_features, marker='o', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Frequency (Hz)')
plt.title('Frequency Across Epochs')
plt.grid(True)
plt.show()

#Frequency Bands:

# Compute average power in alpha and beta bands
alpha_band = (8, 13)
beta_band = (13, 30)

#alpha_power = np.mean(power_spectral_density[:, (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])], axis=1)
#beta_power = np.mean(power_spectral_density[:, (freqs >= beta_band[0]) & (freqs <= beta_band[1])], axis=1)

# Create a stacked area plot
plt.figure(figsize=(10, 6))
plt.stackplot(range(len(alpha_power)), alpha_power, beta_power, labels=['Alpha', 'Beta'], alpha=0.7)
plt.xlabel('Epochs')
plt.ylabel('Power')
plt.title('Power in Alpha and Beta Bands')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()


# Plot KFD values
plt.figure(figsize=(10, 6))
plt.plot(range(len(kfd_values)), kfd_values, marker='o', color='blue')
plt.xlabel('Epochs')
plt.ylabel('KFD')
plt.title('KFD Across Epochs')
plt.grid(True)
plt.show()

# Plot Shannon entropy values
plt.figure(figsize=(10, 6))
plt.plot(range(len(entropy_values)), entropy_values, marker='o', color='green')
plt.xlabel('Epochs')
plt.ylabel('Shannon Entropy')
plt.title('Shannon Entropy Across Epochs')
plt.grid(True)
plt.show()
