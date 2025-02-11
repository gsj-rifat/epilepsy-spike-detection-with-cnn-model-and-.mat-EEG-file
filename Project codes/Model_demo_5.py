import glob
import os
import re

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow
import tensorflow as tf
from scipy.io import loadmat
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (BatchNormalization, Conv1D, Conv2D, Dense,
                                     Dropout, Flatten, MaxPooling1D,
                                     MaxPooling2D, Reshape)
from tensorflow.keras.models import Sequential
from tensorflow.math import count_nonzero
from tensorflow.python.client.session import Session
from tensorflow.python.ops.variables import global_variables_initializer


# Sort the files based on their numeric endings
# Define a custom sorting function
def natural_sort_key(filename):
    # Extract the numeric part from the filename
    numeric_part = ''.join(c for c in filename if c.isdigit())
    return int(numeric_part) if numeric_part else filename


# labels structure building
path_to_sub_labels = 'C:\\masterthesis-golam-rifat\\Feature folder\\Spike labels\\Event files time points shift0\\data_marked_by_ep_3\\'
# Retrieve all absolute pathnames to matching subdirectories
path_to_labels = glob.glob(f'{path_to_sub_labels}/*.txt')

combined_labels = {}
times = np.arange(3600)
i = 0
path_to_labels = sorted(path_to_labels, key=natural_sort_key)

for file in path_to_labels:
    # print(file)
    event = np.loadtxt(file)
    event = event.astype(int)
    event = np.unique(event)
    labels = [1 if time in event else 0 for time in times]
    combined_label = ()
    for x in range(0, 19):
        combined_label = np.hstack((combined_label, labels))

    combined_labels[f'combined_label_{i}'] = combined_label
    # print(combined_labels[f'combined_label_{i}'].shape)
    # print(i)
    i = i + 1

# data structure building
# Specify the path to your directory
path_to_spec_feature = 'C:\\masterthesis-golam-rifat\\Feature folder\\Spectro Feature\\'

# Retrieve all absolute pathnames to matching subdirectories
path_to_spec = glob.glob(f'{path_to_spec_feature}/**/')  # *.csv

combined_spec_dfs = {}
spec_num = 0
# Specify the path to your directory containing CSV files
for path in path_to_spec:
    print(path)
    files = glob.glob(f'{path}/*.csv')  # *.csv
    combined_spec_df = pd.DataFrame()
    for file in files:
        spec_df = pd.read_csv(file)
        spec_df= (spec_df - spec_df.min()) / (spec_df.max() - spec_df.min())

        # print(file, spec_df.shape)
        # print(spec_df.iloc[:64,1:28801])
        spec_df = spec_df.iloc[:64, 1:28801]
        combined_spec_df = pd.concat([combined_spec_df, spec_df], axis=1)
    combined_spec_dfs[f'combined_spec_df_{spec_num}'] = combined_spec_df
    # print('\nspectro\n', combined_spec_dfs[f'combined_spec_df_{spec_num}'].shape)
    # print('\nspectro\n', combined_spec_dfs[f'combined_spec_df_{spec_num}'])
    spec_num = spec_num + 1

# Split the data and label and save in dictionary
X_train = {}
X_test = {}
Y_train = {}
Y_test = {}
X_val = {}
X_pred = {}
Y_val = {}
Y_pred = {}
x_train = ()
x_test = ()
y_train = ()
y_test = ()
x_val = ()
x_pred = ()
y_val = ()
y_pred = ()

# Split the data into train and test sets
total_seconds = 68400
time_points_per_second = 8
frequency_bins = 64
for j in range(0, 22):
    # print(combined_spec_dfs[f'combined_spec_df_{j}'].shape)
    # print(combined_spec_dfs[f'combined_spec_df_{j}'])
    # Reshape the data to match seconds
    combined_spec_dfs[f'combined_spec_df_{j}'] = combined_spec_dfs[f'combined_spec_df_{j}'].values.reshape(
        total_seconds, time_points_per_second, frequency_bins)

    # print(combined_spec_dfs[f'combined_spec_df_{j}'])

    # print(combined_spec_dfs[f'combined_spec_df_{j}'].shape, combined_labels[f'combined_label_{j}'].shape)

    X_train[f'x_train_{j}'], X_test[f'x_test_{j}'], Y_train[f'y_train_{j}'], Y_test[f'y_test_{j}'] = train_test_split(
        combined_spec_dfs[f'combined_spec_df_{j}'], combined_labels[f'combined_label_{j}'], test_size=0.3,
        random_state=42)

    X_val[f'x_val_{j}'], X_pred[f'x_pred_{j}'], Y_val[f'y_val_{j}'], Y_pred[f'y_pred_{j}'] = train_test_split(
        X_test[f'x_test_{j}'], Y_test[f'y_test_{j}'], test_size=0.5, random_state=42)

    y_train = np.hstack((y_train, Y_train[f'y_train_{j}']))
    # x_val = np.concatenate((x_val, X_val[f'x_val_{i}']), axis=0)
    # x_pred = np.concatenate((x_pred, X_pred[f'x_pred_{i}']), axis=0)
    y_val = np.hstack((y_val, Y_val[f'y_val_{j}']))
    y_pred = np.hstack((y_pred, Y_pred[f'y_pred_{j}']))
    print(X_train[f'x_train_{j}'].shape, X_val[f'x_val_{j}'].shape, X_pred[f'x_pred_{j}'].shape, y_train.shape,
          y_pred.shape, y_val.shape)

x_train = np.concatenate(list(X_train.values()), axis=0)
x_train = x_train.reshape(-1, 1, 8, 64)
x_val = np.concatenate(list(X_val.values()), axis=0)
x_val = x_val.reshape(-1, 1, 8, 64)
x_pred = np.concatenate(list(X_pred.values()), axis=0)
x_pred = x_pred.reshape(-1, 1, 8, 64)
print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)
# print(x_train)



def relative_accuracy(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))

    # Calculate the relative accuracy
    relative_accuracy = (true_positives / (possible_positives + K.epsilon())) + \
                        (true_negatives / (possible_negatives + K.epsilon())) / 2

    return relative_accuracy



callback_functions = [
    tensorflow.keras.callbacks.EarlyStopping(
        monitor='relative_accuracy',
        patience=20,
        verbose=1
    ),
    tensorflow.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        patience=15,
        verbose=1,
        factor=0.5,
        min_lr=0.00001
    )
]


#seq_model = tensorflow.keras.Input(shape=10)
# Define the CNN model
model = Sequential()
#model.add(Reshape((8, 64, 1), input_shape=(1, 8, 64)))
model.add(Conv2D(filters=128, padding='same', kernel_size=(3, 3), activation='relu', input_shape=(1, 8, 64)))
BatchNormalization()
model.add(MaxPooling2D((1, 4)))
model.add(Conv2D(filters=1024, padding='same', kernel_size=(4, 4), activation='relu'))
BatchNormalization()
model.add(MaxPooling2D((1, 1)))
model.add(Flatten())
#model.add(MaxPooling2D((4, 4)))
#model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Binary classification (spike or non-spike)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=relative_accuracy)

# Print summary of the model architecture
model.summary()
class_weights = {0: 1, 1: 10}

# Train the model
history = model.fit(x_train, y_train, epochs=500, batch_size=64, verbose=2, validation_data=(x_val, y_val),
                    callbacks=callback_functions, class_weight=class_weights)

loss, accuracy = model.evaluate(x_pred, y_pred, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

# Get predictions
y_test = model.predict(x_pred)
y_test = [1 if prob > 0.5 else 0 for prob in y_test]
# Compute confusion matrix
cm = confusion_matrix(y_pred, y_test)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print("Confusion Matrix:")
print(cm)


plt.plot(history.history['relative_accuracy'], label='rel_accuracy')
plt.plot(history.history['val_relative_accuracy'], label='val_rel_accuracy')
plt.plot(history.history['loss'], label='loss')
#plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Rel_Accuracy')
#plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
sns.heatmap(cm, annot=True, fmt='d')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

positive_precision = precision_score(y_test, y_pred, pos_label=1)
print(f"Positive Precision: {positive_precision:.2f}")
positive_recall = recall_score(y_test, y_pred, pos_label=1)
print(f"Positive Recall: {positive_recall:.2f}")
f1_score = 2 * (positive_precision * positive_recall) / (positive_precision + positive_recall)
print(f"F1 Score: {f1_score:.2f}")


