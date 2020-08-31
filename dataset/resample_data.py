
# packages
import numpy as np
import pandas as pd
import scipy as sy
from scipy import signal


def load_data(filename: str, label: int):
    '''
        filename - is the file to load data from

        label - what to label windows from this file as an int
            0:'move_up',
            1:'move_down',
            2:'no_movement'
    '''
    # load the data
    data = np.load(f'{filename}')

    # label the columns
    df = pd.DataFrame({
        'time': data[:, 0],
        'emg1': data[:, 1],
        'emg2': data[:, 2],
        'pos_v': data[:, 3]
        })

    df['time'] = pd.TimedeltaIndex(df['time'].values, 's')
    df = df.resample('1L', on='time').mean()

    sos = signal.butter(4, 20, 'hp', fs=1000, output='sos')
    df['emg1'] = signal.sosfilt(sos, df['emg1'])
    df['emg2'] = signal.sosfilt(sos, df['emg2'])

    # the first 60 samples are dropped due to noise
    df['emg1'] = df['emg1'][60:]
    df['emg2'] = df['emg2'][60:]
    df = df.dropna()

    # create a list of labels which has a label for each window
    labels = [label for d in data]

    return df, labels


if __name__ == "__main__":
    input_path = "./dataset/raw"
    output_path = "./dataset/filtered"

    # patient 1
    for i in range(1, 16):
        data, labels = load_data(f'{input_path}/move_up/1_{i}.npy', 0)
        arr = data.to_numpy()
        np.save(f'{output_path}/move_up/1_{i}', arr)

    for i in range(1, 20):
        data, labels = load_data(f'{input_path}/move_down/1_{i}.npy', 1)
        arr = data.to_numpy()
        np.save(f'{output_path}/move_down/1_{i}', arr)

    for i in range(1, 3):
        data, labels = load_data(f'{input_path}/no_movement/1_{i}.npy', 2)
        arr = data.to_numpy()
        np.save(f'{output_path}/no_movement/1_{i}', arr)

    # patient 2
    for i in range(1, 16):
        data, labels = load_data(f'{input_path}/move_up/2_{i}.npy', 0)
        arr = data.to_numpy()
        np.save(f'{output_path}/move_up/2_{i}', arr)

    for i in range(1, 16):
        data, labels = load_data(f'{input_path}/move_down/2_{i}.npy', 1)
        arr = data.to_numpy()
        np.save(f'{output_path}/move_down/2_{i}', arr)

    for i in range(1, 14):
        data, labels = load_data(f'{input_path}/no_movement/2_{i}.npy', 2)
        arr = data.to_numpy()
        np.save(f'{output_path}/no_movement/2_{i}', arr)

    # patient 3
    for i in range(1, 15):
        data, labels = load_data(f'{input_path}/move_up/3_{i}.npy', 0)
        arr = data.to_numpy()
        np.save(f'{output_path}/move_up/3_{i}', arr)

    for i in range(1, 18):
        data, labels = load_data(f'{input_path}/move_down/3_{i}.npy', 1)
        arr = data.to_numpy()
        np.save(f'{output_path}/move_down/3_{i}', arr)

    for i in range(1, 7):
        data, labels = load_data(f'{input_path}/no_movement/3_{i}.npy', 2)
        arr = data.to_numpy()
        np.save(f'{output_path}/no_movement/3_{i}', arr)

    # patient 4
    for i in range(1, 21):
        data, labels = load_data(f'{input_path}/move_up/4_{i}.npy', 0)
        arr = data.to_numpy()
        np.save(f'{output_path}/move_up/4_{i}', arr)

    for i in range(1, 21):
        data, labels = load_data(f'{input_path}/move_down/4_{i}.npy', 1)
        arr = data.to_numpy()
        np.save(f'{output_path}/move_down/4_{i}', arr)

    for i in range(1, 11):
        data, labels = load_data(f'{input_path}/no_movement/4_{i}.npy', 2)
        arr = data.to_numpy()
        np.save(f'{output_path}/no_movement/4_{i}', arr)
