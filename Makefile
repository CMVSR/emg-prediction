init:
	make build
	make create_dataset

## BUILD DOCKER IMAGES

build:
	bash ./scripts/build.sh

## CREATE DATASETS

create_dataset:
	make create_filtered
	make train_model
	make feature_extraction

create_filtered:
	bash ./scripts/create_filtered.sh

train_model:
	bash ./scripts/train_model.sh

feature_extraction:
	bash ./scripts/feature_extraction.sh

## RUN RL ALGORITHM

run:
	bash ./scripts/run_one.sh

run_many:
	bash ./scripts/run_many.sh

debug:
	bash ./scripts/debug.sh

test:
	bash ./scripts/test.sh