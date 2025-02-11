#this is the final model and everything is done here in one file. You can save all the data in a file for training and and test for sequential and CNN model and just use the model code just in case.

import glob
import os
import re

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow
from scipy.io import loadmat
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (BatchNormalization, Conv1D, Conv2D, Dense,
                                     Dropout, Flatten, MaxPooling1D,
                                     MaxPooling2D, Reshape, concatenate, Input)
from tensorflow.keras.models import Model, Sequential
from tensorflow.math import count_nonzero


# Sort the files based on their numeric endings
# Define a custom sorting function
def natural_sort_key(filename):
    # Extract the numeric part from the filename
    numeric_part = ''.join(c for c in filename if c.isdigit())
    return int(numeric_part) if numeric_part else filename


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
        patience=15,
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

# spectrogram/CNN part
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
path_to_spec = sorted(path_to_spec, key=natural_sort_key)
combined_spec_dfs = {}
spec_num = 0
# Specify the path to your directory containing CSV files
for path in path_to_spec:
    subfolder_spec = glob.glob(f'{path}/**/')
    subfolder_spec = sorted(subfolder_spec, key=natural_sort_key)
    print(subfolder_spec)
    for sub in subfolder_spec:
        print(sub)
        files = glob.glob(f'{sub}/*.csv')  # *.csv
        files = sorted(files, key=natural_sort_key)
        combined_spec_df = pd.DataFrame()
        for file in files:
            print(files)
            spec_df = pd.read_csv(file)
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
for j in range(0, 88):
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
# print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)
# print(x_train)


# sequential part
# data structure building
# Specify the path to your directory
path_to_dir = 'C:\\masterthesis-golam-rifat\\Feature folder\\Channel feature\\'

# Retrieve all absolute pathnames to matching subdirectories
path_to_data = glob.glob(f'{path_to_dir}/**/')  # *.csv
path_to_data = sorted(path_to_data, key=natural_sort_key)
combined_dfs = {}
# Specify the path to your directory containing CSV files
df_num = 0
for path in path_to_data:
    subfolder = glob.glob(f'{path}/**/')  # *.csv
    # print(subfolder)
    for file in range(0, 22):
        # print(subfolder[file])
        csv_files = glob.glob(os.path.join(subfolder[file], '*.csv'))
        # print(subfolder[file])
        combined_df = pd.DataFrame()
        for csv_file in csv_files:
            # print(csv_file)
            df = pd.read_csv(csv_file)
            df = df.iloc[:3600, 1:]
            # combined_df = pd.concat([temp_df,df], axis=1) #[f'combined_df_{file}']
            # combined_df._append(df)
            combined_df = pd.concat([combined_df, df], axis=1)
        # print(combined_df)
        # combined_dfs = {}
        combined_dfs[f'combined_df_{df_num}'] = combined_df
        df_num = df_num + 1
        # print('\nfeature combined_dfs[file]\n',combined_dfs[f'combined_df_{file}'].shape)

# for i in combined_dfs:
# print(i)
# print(combined_dfs[i])


# labels structure building
path_to_sub_labels = 'C:\\masterthesis-golam-rifat\\Feature folder\\Spike labels\\Event files per neurologist\\Neuro 3\\'
# Retrieve all absolute pathnames to matching subdirectories
path_to_labels = glob.glob(f'{path_to_sub_labels}/**/')

combined_labels = {}
times = np.arange(3600)
i = 0

for sub in path_to_labels:
    txt_files = glob.glob(os.path.join(sub, '*.txt'))
    # Sort the files based on the custom key
    txt_files = sorted(txt_files, key=natural_sort_key)
    # Now 'txt_files' contains the file paths in natural order
    # print(sub)
    for file in txt_files:
        # print(file)
        event = np.loadtxt(file)
        event = event.astype(int)
        event = np.unique(event)
        labels = [1 if time in event else 0 for time in times]
        combined_label = ()
        combined_label = np.hstack((combined_label, labels))
        combined_labels[f'combined_label_{i}'] = combined_label
        # print(combined_labels[f'combined_label_{i}'])
        i = i + 1

# for y in combined_labels:
# print(y)

# Split the data and label and save in dictionary
X_train_seq = {}
X_test_seq = {}
Y_train_seq = {}
Y_test_seq = {}
X_val_seq = {}
X_pred_seq = {}
Y_val_seq = {}
Y_pred_seq = {}
x_train_seq = ()
x_test_seq = ()
y_train_seq = ()
y_test_seq = ()
x_val_seq = ()
x_pred_seq = ()
y_val_seq = ()
y_pred_seq = ()

for i in range(0, 22):
    X_train_seq[f'x_train_{i}'], X_test_seq[f'x_test_{i}'], Y_train_seq[f'y_train_{i}'], Y_test_seq[
        f'y_test_{i}'] = train_test_split(
        combined_dfs[f'combined_df_{i}'], combined_labels[f'combined_label_{i}'], test_size=0.3, random_state=42)
    X_val_seq[f'x_val_{i}'], X_pred_seq[f'x_pred_{i}'], Y_val_seq[f'y_val_{i}'], Y_pred_seq[
        f'y_pred_{i}'] = train_test_split(
        X_test_seq[f'x_test_{i}'], Y_test_seq[f'y_test_{i}'], test_size=0.5, random_state=42)
    y_train_seq = np.hstack((y_train_seq, Y_train_seq[f'y_train_{i}']))
    # x_val_seq = np.concatenate((x_val_seq, X_val_seq[f'x_val_{i}']), axis=0)
    # x_pred_seq = np.concatenate((x_pred_seq, X_pred_seq[f'x_pred_{i}']), axis=0)
    y_val_seq = np.hstack((y_val_seq, Y_val_seq[f'y_val_{i}']))
    y_pred_seq = np.hstack((y_pred_seq, Y_pred_seq[f'y_pred_{i}']))
    # print(X_train_seq[f'x_train_{i}'].shape, X_val_seq[f'x_val_{i}'].shape, X_pred_seq[f'x_pred_{i}'].shape)


#reshape the array
x_train_seq = np.concatenate(list(X_train_seq.values()), axis=0)
x_train_seq = x_train_seq.reshape(1053360, 10)
x_val_seq = np.concatenate(list(X_val_seq.values()), axis=0)
x_val_seq = x_val_seq.reshape(225720, 10)
x_pred_seq = np.concatenate(list(X_pred_seq.values()), axis=0)
x_pred_seq = x_pred_seq.reshape(225720, 10)
print(x_train_seq.shape, x_val_seq.shape, y_train_seq.shape, y_val_seq.shape)



# Define the CNN model
cnn_model = Sequential()
cnn_model.add(Conv2D(filters=128, padding='same', kernel_size=(3, 3), activation='relu', input_shape=(1, 8, 64)))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D((1, 4)))
cnn_model.add(Conv2D(filters=1024, padding='same', kernel_size=(4, 4), activation='relu'))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D((1, 1)))
cnn_model.add(Flatten())
cnn_model.add(Dense(512, activation='relu'))

# Define the Sequential model
seq_model = Sequential()
seq_model.add(Dense(10, input_dim=10, activation='relu'))
seq_model.add(Dense(256, activation='relu'))
seq_model.add(Dense(512, activation='relu'))
seq_model.add(Dense(1024, activation='relu'))
seq_model.add(Dropout(0.2))
seq_model.add(Dense(1024, activation='relu'))
seq_model.add(Dropout(0.2))
seq_model.add(Dense(512, activation='relu'))
seq_model.add(Flatten())

# Create the models
cnn_model = tensorflow.keras.Model(inputs=cnn_model.input, outputs=cnn_model.output)
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=relative_accuracy)
# cnn_model.fit(x_train, y_train)
# pred1 = cnn_model.predict(x_val)
# print(pred1.shape)

seq_model = tensorflow.keras.Model(inputs=seq_model.input, outputs=seq_model.output)
seq_model.compile(optimizer='adam', loss='binary_crossentropy', metrics= relative_accuracy)
# seq_model.fit(x_train_seq, y_train_seq)
# pred2 = seq_model.predict(x_val_seq)
# print(pred2.shape)
# print((y_val.shape, y_val_seq.shape))

# combined_pred = np.row_stack((pred1, pred2))
# y_val = y_val.reshape(-1, 1)
# y_val_seq = y_val_seq.reshape(-1, 1)
# y_new_val = np.row_stack((y_val, y_val_seq))
# print(combined_pred.shape, y_new_val.shape)

# Concatenate the output layers
combined_features = concatenate([cnn_model.output, seq_model.output])
# Additional layers
combined_features = Dense(1024, activation='relu')(combined_features)
output_layer = Dense(1, activation='sigmoid')(combined_features)

# Create the combined model
combined_model = Model(inputs=[cnn_model.input, seq_model.input], outputs=output_layer)
combined_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=relative_accuracy)

print(combined_model.summary())
class_weights = {0: 1, 1: 5}

history = combined_model.fit(
    [x_train, x_train_seq],  # Pass both input arrays
    y_train,
    epochs=500,
    validation_data=([x_val, x_val_seq], y_val),
    callbacks=callback_functions,
    class_weight=class_weights,
    verbose=2
)

loss, accuracy = combined_model.evaluate([x_pred, x_pred_seq], y_pred, verbose=2)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

# Get predictions
y_test = combined_model.predict([x_pred, x_pred_seq])
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
#sns.heatmap(cm, annot=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

positive_precision = precision_score(y_test, y_pred, pos_label=1)
print(f"Positive Precision: {positive_precision:.2f}")
positive_recall = recall_score(y_test, y_pred, pos_label=1)
print(f"Positive Recall: {positive_recall:.2f}")
f1_score = 2 * (positive_precision * positive_recall) / (positive_precision + positive_recall)
print(f"F1 Score: {f1_score:.2f}")









