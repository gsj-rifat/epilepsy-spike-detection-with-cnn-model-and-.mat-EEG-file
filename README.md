# epilepsy-spike-detection-with-cnn-model-and-.mat-EEG-file
Automatic Detection of Epilepsy-typical  Spikes in EEG with CNN and Sequential model.
In this project, the data is a multichannel EEG data but it is a 2D array with .mat format. So the data is converted into Numpy array, on the other side the metadata is creacted.
Then the raw data is produced with joining the metadata and the numpy array. Then the data is filtered with low and highpass and notch filter.
If you have more than one files of data, the you can also concatenate the data and build the raw data.
There are one set of data, but for more data requirements, three sets of synthetic data is produced by time shifting method.
Then the epochs are created and features are extracted. Features are Mean Amplitude Value,Amplitude Standard Deviation, Amplitude Skewness, Amplitude Kurtosis, Katz Fractal Dimension, Shannon’s Entropy and Spectrogram.

After that the model is build. About the model, it is an Ensemble model because it is created with both CNN and sequential model. The base work is the spectrogram is processed in the CNN model first and then the output is fed to a sequential model along with the other features.

