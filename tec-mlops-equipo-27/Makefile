#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = tec-mlops-equipo-27
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format





## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data:
	$(PYTHON_INTERPRETER) -m mibici.dataset

## Build hourly panel and features from verified data
.PHONY: features
features:
	$(PYTHON_INTERPRETER) -m mibici.preparar

## Execute baselines P1, P2 and P3 and log runs in MLflow
.PHONY: baseline
baseline:
	$(PYTHON_INTERPRETER) -m mibici.modeling.train

## Open local MLflow Tracking UI (http://127.0.0.1:5000)
.PHONY: tracking
tracking:
	$(PYTHON_INTERPRETER) -m mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000

## Run reproducible tests without downloading MiBici files
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m unittest discover -s tests -v


## Check Python code and tracked/pending files for security issues
.PHONY: security security-code security-secrets
security: security-code security-secrets

## Scan application code and security scripts with Bandit
security-code:
	$(PYTHON_INTERPRETER) -m bandit -r mibici scripts

## Reject new potential secrets without sending them to external services
security-secrets:
	$(PYTHON_INTERPRETER) scripts/check_secrets.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
