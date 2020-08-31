# emg-prediction

This project uses an A2C algorithm to classify sEMG signals as move up or move down.

# Running

`Docker` is used to run the app, and `make` is used to make installation, building, and running the project simpler. The models are run with tensorflow on gpus. Since GPU support is not enable for `docker-compose`, scripts and dockerfiles are used to support the project instead.

`make init` will build the datasets required for training the RL model.

`make run` will run the algorithm once on the generated dataset and store the results in `/logs`.

# Development



# Cite the paper

`todo`