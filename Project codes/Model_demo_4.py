import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Reshape
import mne
from tensorflow.keras import backend as K
import seaborn as sns
import numpy as np
import pandas as pd
import glob
import os
import re
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix

# data structure building

# Specify the path to your directory
path_to_dir = 'C:\\masterthesis-golam-rifat\\Feature folder\\Channel feature\\'

# Retrieve all absolute pathnames to matching subdirectories
path_to_data = glob.glob(f'{path_to_dir}/**/')  # *.csv

combined_dfs = {}
# Specify the path to your directory containing CSV files
df_num = 0
for path in path_to_data:
    subfolder = glob.glob(f'{path}/**/')  # *.csv
    print(subfolder)
    for file in range(0, 22):
        print(subfolder[file])
        csv_files = glob.glob(os.path.join(subfolder[file], '*.csv'))
        # print(subfolder[file])
        combined_df = pd.DataFrame()
        for csv_file in csv_files:
            print(csv_file)
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

# Sort the files based on their numeric endings
# Define a custom sorting function
def natural_sort_key(filename):
    # Extract the numeric part from the filename
    numeric_part = ''.join(c for c in filename if c.isdigit())
    return int(numeric_part) if numeric_part else filename


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
    print(sub)
    for file in txt_files:
        print(file)
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

for i in range(0, 87):
    X_train[f'x_train_{i}'], X_test[f'x_test_{i}'], Y_train[f'y_train_{i}'], Y_test[f'y_test_{i}'] = train_test_split(
        combined_dfs[f'combined_df_{i}'], combined_labels[f'combined_label_{i}'], test_size=0.3, random_state=42)
    X_val[f'x_val_{i}'], X_pred[f'x_pred_{i}'], Y_val[f'y_val_{i}'], Y_pred[f'y_pred_{i}'] = train_test_split(
        X_test[f'x_test_{i}'], Y_test[f'y_test_{i}'], test_size=0.5, random_state=42)
    y_train = np.hstack((y_train, Y_train[f'y_train_{i}']))
    # x_val = np.concatenate((x_val, X_val[f'x_val_{i}']), axis=0)
    # x_pred = np.concatenate((x_pred, X_pred[f'x_pred_{i}']), axis=0)
    y_val = np.hstack((y_val, Y_val[f'y_val_{i}']))
    y_pred = np.hstack((y_pred, Y_pred[f'y_pred_{i}']))
    # print(X_train[f'x_train_{i}'].shape, X_val[f'x_val_{i}'].shape, X_pred[f'x_pred_{i}'].shape)

x_train = np.concatenate(list(X_train.values()), axis=0)

x_val = np.concatenate(list(X_val.values()), axis=0)

x_pred = np.concatenate(list(X_pred.values()), axis=0)

# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

def relative_accuracy(y_true, y_pred):
    # Calculate True Positives (TP) and True Negatives (TN)
    y_true_rounded = K.round(y_true)
    y_pred_rounded = K.round(y_pred)
    tp = K.sum(y_true_rounded * y_pred_rounded)
    tn = K.sum((1 - y_true_rounded) * (1 - y_pred_rounded))

    # Calculate total number of samples
    total_samples = K.cast(K.shape(y_test)[0], dtype='float32')

    # Calculate relative accuracy
    relative_accuracy = tn / (tn + K.sum(y_true_rounded))
    return relative_accuracy




callback_functions = [
    tensorflow.keras.callbacks.EarlyStopping(
        monitor='relative_accuracy',
        patience=15,
        verbose=2
    ),
    tensorflow.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        patience=15,
        verbose=2,
        factor=0.5,
        min_lr=0.00001
    )
]

# Define the model
model = Sequential()
# Input layer
model.add(Dense(190, input_dim=190, activation='relu'))
# Hidden layers
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))  # tanh, relu, sigmoid
# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
model.add(Dense(1024, activation='relu'))
# model.add(Dense(512, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))

# model.add(Dropout(0.2))
# model.add(Flatten())
# Output layer
model.add(Dense(1, activation='sigmoid'))  # softmax sigmoid

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=relative_accuracy)  # sparse_categorical_crossentropy, binary_crossentropy

# Print summary of the model architecture
model.summary()

# Define class weights
class_weights = {0: 1, 1: 5}

# Train the model
history = model.fit(x_train, y_train, epochs=300, batch_size=32, validation_data=(x_val, y_val),
                    callbacks=callback_functions, class_weight= class_weights,
                    verbose=2)  # callbacks= callback_functions,
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
sns.heatmap(cm, annot=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

positive_precision = precision_score(y_test, y_pred, pos_label=1)
print(f"Positive Precision: {positive_precision:.2f}")
positive_recall = recall_score(y_test, y_pred, pos_label=1)
print(f"Positive Recall: {positive_recall:.2f}")
f1_score = 2 * (positive_precision * positive_recall) / (positive_precision + positive_recall)
print(f"F1 Score: {f1_score:.2f}")





