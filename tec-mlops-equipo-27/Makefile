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

## Serve the P1 demand model as a REST API (FastAPI) on port 8000
.PHONY: api
api:
	$(PYTHON_INTERPRETER) -m uvicorn mibici.api.app:app --host 0.0.0.0 --port 8000

## Validate panel quality, integrity and train/test drift with DeepChecks
.PHONY: validate
validate:
	$(PYTHON_INTERPRETER) -m mibici.validacion

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


# Orquestación (docker-compose.yml): usa podman-compose si existe, si no docker compose
COMPOSE ?= $(shell if command -v podman-compose >/dev/null 2>&1; then echo podman-compose; else echo docker compose; fi)
COMPOSE_FILE = docker-compose.yml

## Download the P1 model from the local registry (sqlite:///mlflow.db) into modelo_p1/
.PHONY: modelo
modelo:
	MLFLOW_TRACKING_URI=$${MLFLOW_TRACKING_URI:-sqlite:///mlflow.db} $(PYTHON_INTERPRETER) -m mibici.api.descargar --destino modelo_p1

## Generate a synthetic DEMO model in modelo_p1/ (clean clone / CI, not the real model)
.PHONY: modelo-demo
modelo-demo:
	$(PYTHON_INTERPRETER) scripts/modelo_demo.py --destino modelo_p1

## Prepare modelo_p1/ if missing and start MLflow + API with one command
.PHONY: up
up:
	@if [ ! -f modelo_p1/MLmodel ]; then \
		if [ -f mlflow.db ]; then \
			echo ">>> modelo_p1/ no existe: se descarga del registry local (mlflow.db)"; \
			$(MAKE) modelo; \
		else \
			echo ">>> AVISO: sin modelo_p1/ ni mlflow.db; se usa un modelo de DEMOSTRACIÓN"; \
			$(MAKE) modelo-demo; \
		fi; \
	fi
	$(COMPOSE) -f $(COMPOSE_FILE) up --build -d

## Stop the services (the mlflow_store volume is kept)
.PHONY: down
down:
	$(COMPOSE) -f $(COMPOSE_FILE) down

## Show service status
.PHONY: ps
ps:
	$(COMPOSE) -f $(COMPOSE_FILE) ps

## Follow service logs
.PHONY: logs
logs:
	$(COMPOSE) -f $(COMPOSE_FILE) logs -f

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
