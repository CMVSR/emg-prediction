
# standard
import random
from datetime import datetime, timedelta, timezone

# packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models


LABELS_DICT = {
    0: 'move_up',
    1: 'move_down',
    2: 'no_movement'
}

WINDOW_SIZE = 250
WINDOW_OVERLAP = WINDOW_SIZE - 100


def load_data(filename: str, label: int) -> (list, list):
    '''
        filename - is the file to load data from

        label - what to label windows from this file as an int
            0:'move_up',
            1:'move_down',
            2:'no_movement'
    '''
    # load the data
    data = np.load(f'{filename}')

    df = pd.DataFrame({
        'emg1': data[:, 0],
        'emg2': data[:, 1],
        'pos_v': data[:, 2]
        })

    # window the data, store in the list called 'windows'
    windows = [df[i:i+WINDOW_SIZE] for i in range(0, len(df), WINDOW_OVERLAP)]

    # delete any windows that do not match the specified size
    for i in range(len(windows)-1, 0, -1):
        if len(windows[i]) != WINDOW_SIZE:
            del windows[i]

    # create a list of labels which has a label for each window
    labels = [label for window in windows]

    if len(windows) != len(labels):
        raise ValueError('number of windows and labels must be equivalent')

    return windows, labels


def window_df_to_array(window: pd.DataFrame) -> np.array:
    # load the data from the input data frame to separate numpy arrays
    window_pos = np.array(window.pos_v)
    window_emg1_data = np.array(window.emg1)
    window_emg2_data = np.array(window.emg2)
    window_emg1_fft = np.fft.fftn(window_emg1_data)
    window_emg2_fft = np.fft.fftn(window_emg2_data)

    # create input array for cnn
    emg_data = np.asarray((
        window_emg1_data,
        window_emg2_data,
        window_emg1_fft,
        window_emg2_fft))

    # reshape the data to fit the network input shape
    res = np.reshape(emg_data, (25, 40, 1))

    return res


def train_model(windows: list, labels: list, log: bool=True):

    # there must be an equal amount of windows and labels
    if len(windows) != len(labels):
        raise ValueError('number of windows and labels must be equivalent')
    emg_data = np.array([window_df_to_array(window) for window in windows])

    print(f'shape of the data before train/test split: {emg_data.shape}')

    # split train and test data
    x_train, x_test, y_train, y_test = train_test_split(
        emg_data, labels, test_size=0.3)

    # define the model architecture
    model = models.Sequential([
        layers.Conv2D(8, (3, 3), activation='relu', input_shape=(25, 40, 1)),
        layers.AveragePooling2D((3, 3)),
        layers.BatchNormalization(),
        # layers.Conv2D(8, (3, 3), activation='relu'),
        # layers.AveragePooling2D((3, 3)),
        # layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])

    # provide a summary of the model
    model.summary()

    # compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # train the model on the data and labels
    print('training model...')
    history = model.fit(x_train, y_train, epochs=50)

    # test the model on the test portion of the data
    print('evaluating model...')
    model.evaluate(x_test, y_test)

    # store the model
    model.save('./models/extraction_model')


if __name__ == "__main__":
    base_path = "./dataset/filtered"

    # load each dataset and their labels
    mu_windows, mu_labels = load_data(
        f'{base_path}/move_up.npy', 0)
    md_windows, md_labels = load_data(
        f'{base_path}/move_down.npy', 1)
    nm_windows, nm_labels = load_data(
        f'{base_path}/no_movement.npy', 2)

    # concatenate data
    windows = mu_windows + md_windows
    labels = np.asarray(mu_labels + md_labels)

    with tf.device('/gpu:0'):
        train_model(windows, labels)

    # test that model saved correctly
    model = models.load_model('./models/extraction_model')
