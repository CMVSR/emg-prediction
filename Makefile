init:
	build
	create_dataset

## BUILD DOCKER IMAGES

build:
	build_app
	build_debug
	build_extraction_model
	build_feature_extraction

build_app:
	bash ./scripts/build_app.sh

build_debug:
	bash ./scripts/build_debug.sh

build_extraction_model:
	bash ./scripts/build_extraction_model.sh

build_feature_extraction:
	bash ./scripts/build_feature_extraction.sh

## CREATE DATASETS

create_dataset:
	create_filtered
	train_model
	feature_extraction

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