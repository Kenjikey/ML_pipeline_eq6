#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = ML_pipeline_eq6
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: setup requirements
setup: requirements

requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	
## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using flake8, black, and isort (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 --max-line-length=99 mibici tests
	isort --check --diff mibici tests
	black --check mibici tests

## Format source code with black
.PHONY: format
format:
	isort mibici tests
	black mibici tests

## Run tests
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest -q

## Launch MLflow tracking UI
.PHONY: mlflow_ui
mlflow_ui:
	mlflow ui --host 127.0.0.1 --port 5001

## Run FastAPI serving locally
.PHONY: api serve
api: serve
serve:
	uvicorn mibici.serving.app:app --host 0.0.0.0 --port 8000 --reload

## Build Docker container for serving
.PHONY: docker-build
docker-build:
	docker build -t mibici-ml-service:0.1.0 .

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Make dataset
.PHONY: data
data:
	$(PYTHON_INTERPRETER) mibici/dataset.py

## Generate features from hourly balance
.PHONY: features
features:
	$(PYTHON_INTERPRETER) mibici/features.py

## Train consolidated pipeline and register in MLflow Model Registry
.PHONY: train
train:
	$(PYTHON_INTERPRETER) mibici/modeling/train.py

## Run batch inference on test split
.PHONY: predict
predict:
	$(PYTHON_INTERPRETER) mibici/modeling/predict.py

## Run drift and performance degradation monitor
.PHONY: monitor
monitor:
	$(PYTHON_INTERPRETER) mibici/monitoring/drift.py