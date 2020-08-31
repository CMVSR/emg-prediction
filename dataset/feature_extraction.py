
# standard
import random

# packages
import pandas as pd
import numpy as np
import tensorflow as tf

# internal
from train_model import load_data, window_df_to_array


def balance_datasets(input_data: list, input_labels: list):
    min_len = len(input_data[0])
    for data in input_data:
        if len(data) < min_len:
            min_len = len(data)

    for data_idx in range(len(input_data)):
        while len(input_data[data_idx]) > min_len:
            idx = random.choice(range(len(input_data[data_idx])))
            del input_data[data_idx][idx]
            del input_labels[data_idx][idx]

    for data in input_data:
        print(len(data))

    return input_data, input_labels


def deep_extraction():
    # extract deep features
    model = tf.keras.models.load_model('./models/extraction_model')
    new_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=model.layers[6].output)
    deep_features = new_model.predict_on_batch(windows)

    return deep_features


if __name__ == "__main__":
    filtered_path = './dataset/filtered'
    extracted_path = './dataset/extracted'

    # load each dataset and their labels
    mu_windows, mu_labels = load_data(
        f'{filtered_path}/move_up.npy', 0)
    md_windows, md_labels = load_data(
        f'{filtered_path}/move_down.npy', 1)
    nm_windows, nm_labels = load_data(
        f'{filtered_path}/no_movement.npy', 2)

    windows, labels = balance_datasets(
        [mu_windows, md_windows, nm_windows],
        [mu_labels, md_labels, nm_labels])

    # concatenate data
    windows = windows[0] + windows[1] + windows[2]
    labels = np.asarray(labels[0] + labels[1] + labels[2])

    # form np array
    windows = np.array([window_df_to_array(window) for window in windows])
    np.save(f'{extracted_path}/windows.npy', windows)
    print(windows[1])
    print(windows.shape)

    with tf.device('/gpu:0'):
        deep_features = deep_extraction()

    # transform deep features
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    deep_features = PCA(n_components=4).fit_transform(deep_features)

    # hand extracted features
    rms_feature = np.array(
        [np.sqrt(np.mean(window**2)) for window in windows],
        dtype=np.float64)
    var_feature = np.array(
        [np.var(window) for window in windows],
        dtype=np.float64)
    mav_feature = np.array(
        [np.mean(np.abs(window)) for window in windows],
        dtype=np.float64)

    features = np.column_stack((
        rms_feature,
        var_feature,
        mav_feature
    ))
    features = StandardScaler().fit_transform(features)

    new_dataset = np.column_stack((
        deep_features,
        features,
        np.array(labels)
    ))

    np.save(f'{extracted_path}/extracted_features.npy', new_dataset)
