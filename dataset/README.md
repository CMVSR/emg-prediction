# /dataset

/dataset contains the data collected for the experiment. 

## /raw

/raw contains the data collected from the original 4 patients. The files are formatted as follows:

filename: `{movement type}/{patient number}_{collection number}.npy`

data: `time, emg sensor 1, emg sensor 2, potentiometer`

## /filtered

After running `make create_filtered`, the /filtered directory will be created. It contains the resampled and filtered patient data. The files are formatted as follows:

filename: `{movement type}/{patient number}_{collection number}.npy`

data: `emg sensor 1, emg sensor 2, potentiometer`

Also, there will be three files with all the patient data concatenated: `move_down.npy, move_up.npy, no_movement.npy`.

## /extracted

After running `make train_model` and `make feature_extraction`, the /extracted directory will be created. It contains the deep and hand extracted features.
The `extracted_features.npy` file contains the dataset used for training the RL model. The file is formatted as follows:

filename: `extracted_features.npy`

data: `deep features [0-3], rms_feature, var_feature, mav_feature, label`
