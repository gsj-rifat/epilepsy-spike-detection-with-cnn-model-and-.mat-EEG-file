                            Automatic Detection of Epilepsy-Typical Spikes in EEG

                            
This repository contains the implementation of a machine learning-based approach for the automatic detection of epilepsy-typical spikes in EEG signals. The project aims to enhance the efficiency and accuracy of spike detection, assisting clinicians in identifying epileptogenic zones and improving seizure onset detection.

**Project Overview**
Epilepsy is a neurological disorder affecting millions worldwide. EEG (Electroencephalography) is a crucial tool in epilepsy diagnosis, but manual spike detection is time-consuming and prone to human error. This project develops a machine learning model to automate and improve the detection of epilepsy-related spikes.

**Key Components**
Data Structure & Visualization

Initially the EEG data is a 2D numpy array in .mat format. First a raw data is created by building a metadata for the 2D numpy array. Then a multichannel EEG data is build out of it. Then Raw EEG data processed using the MNE-Python library.
Visualization of EEG signals and comparison between channels.

**Preprocessing**

Noise filtering: Applied 50Hz notch filter and bandpass filtering (0.25Hz–50Hz) to remove artifacts.
Epoching: Divided signals into fixed-length segments with overlapping and non-overlapping strategies.
Data augmentation: Used techniques like noise injection, signal shifting, scaling, cropping, and generative models.

**Feature Extraction**

Extracted statistical, fractal, and entropy-based features.
Spectrogram generation for CNN-based models.

**Model Development & Evaluation**

Implemented Sequential Model, CNN Model, and Ensemble Model.
Evaluated using accuracy, precision, recall, and confusion matrices.
CNN Model achieved 98.62% training accuracy, but showed overfitting.
Sequential Model provided a more reliable detection performance for non-spike cases.
Ensemble Model showed room for improvement in balancing precision and recall.

**Future Work**

Improving data quality and feature selection.
Fine-tuning model parameters and exploring alternative architectures.
Expanding validation with diverse patient datasets.

**Technologies Used**
Python
MNE Library (EEG signal processing)
NumPy, Pandas, Matplotlib (Data processing & visualization)
TensorFlow / PyTorch (Machine learning model development)
Scikit-Learn (Evaluation metrics)
