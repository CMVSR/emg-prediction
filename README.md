# emg-prediction

This project uses an A2C algorithm to classify sEMG signals.

# Running

`Docker` is used to run the app, and `make` is used to make installation, building, and running the project simpler. The models are run with tensorflow on gpus. Since GPU support is not enabled for `docker-compose`, scripts and dockerfiles are used to support the project instead.

1. `make init` will build the datasets required for training the RL model.
2. `make run` will run the algorithm once on the generated dataset and store the results in `/logs`.

# Development

`/app/app.py` is the main entrypoint for the project and is where the RL model is trained and evaluated.

`/app/env.py` defines the environment used by the agent using the `gym` environment template.

## Docker
There are 4 docker images used in the project. `app, debug, extraction_model, and feature_extraction`.
1. `app`: Image used to run the main RL algorithm.
2. `debug`: Image used to run the main RL algorithm. It can be debugged after launching with `make debug` by using `Attach to Docker Run` in vscode.
3. `extraction_model`: Image used to train the CNN for deep feature extraction.
4. `feature_extraction`: Image used to obtain dataset of hand extracted and deep extracted features by using the filtered dataset and trained extraction model.

## Dataset
1. The repo is initialized with the raw data collected from 4 healthy patients. To create the resampled and filtered dataset, run `make create_filtered`.
2. After the resampled and filtered dataset is created, the CNN for deep feature extraction is trained. To train the model, run `make train_model`.
3. Hand extracted and deep extracted features are used to create the dataset for RL training. Run `make feature_extraction`. 


# Cite the paper

See CITATION.bib

```
@inproceedings{gardner_emg_2020,
	title = {{EMG} {Based} {Simultaneous} {Wrist} {Motion} {Prediction} {Using} {Reinforcement} {Learning}},
	doi = {10.1109/BIBE50027.2020.00172},
	author = {Gardner, Noah and Tekes, Coskun and Weinberg, Nate and Ray, Nick and Duran, Julian and Housley, Stephen N. and Wu, David and Hung, Chih-Cheng},
	month = oct,
	year = {2020},
	note = {ISSN: 2471-7819},
	keywords = {Wrist, Performance evaluation, Biological system modeling, Reinforcement learning, Feature extraction, Robot sensing systems, Testing, EMG, Stroke Rehabilitation, CNN, Reinforcement Learning},
	pages = {1016--1021},
}
```
