# Makefile — основные команды проекта. Запускать из папки farm/.
#   make install   зависимости для разработки (Python + Node)
#   make test      pytest + jest
#   make lint      ruff + mypy + node --check
#   make build     Docker-образ генератора (model_server) для Cloud Run Job
#   make deploy    push образа и обновление Cloud Run Job (нужен CONFIRM=yes)

PYTHON   ?= python3
NPM      ?= npm
IMAGE_TAG ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)

# Для build/deploy (можно задать в окружении). Значения по умолчанию — из config.json.
GCP_PROJECT_ID ?= $(shell $(PYTHON) -c "import json;print(json.load(open('config.json'))['gcp'].get('project_id',''))" 2>/dev/null)
GCP_REGION     ?= $(shell $(PYTHON) -c "import json;print(json.load(open('config.json'))['gcp'].get('region','us-central1'))" 2>/dev/null)
GCP_JOB_NAME   ?= $(shell $(PYTHON) -c "import json;print(json.load(open('config.json'))['gcp'].get('job_name','prometheus-generate'))" 2>/dev/null)
AR_REPO        ?= prometheus
IMAGE          ?= $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT_ID)/$(AR_REPO)/model-server:$(IMAGE_TAG)

.PHONY: help install test test-py test-js lint build deploy clean

help:
	@grep -E '^#   make' Makefile | sed 's/^#   //'

install:
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(NPM) ci

test: test-py test-js

test-py:
	$(PYTHON) -m pytest -q

test-js:
	$(NPM) test --silent -- --ci

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m mypy
	$(NPM) run --silent check

build:
	@test -f requirements-gpu.txt || { echo "нет requirements-gpu.txt (его нет в репозитории — см. README, раздел «Чего нет в репозитории»)"; exit 1; }
	docker build -t model-server:$(IMAGE_TAG) .
	@echo "образ: model-server:$(IMAGE_TAG)"

# [ПРОВЕРИТЬ ПО ДОКАМ] флаги gcloud: Cloud Run Job с GPU создаётся заранее (см. README → Деплой);
# здесь только новый образ для существующего Job.
deploy: build
	@test "$(CONFIRM)" = "yes" || { echo "деплой меняет продакшен-Job $(GCP_JOB_NAME) в $(GCP_PROJECT_ID): запусти make deploy CONFIRM=yes"; exit 1; }
	@test -n "$(GCP_PROJECT_ID)" || { echo "не задан GCP_PROJECT_ID"; exit 1; }
	gcloud auth configure-docker $(GCP_REGION)-docker.pkg.dev --quiet
	docker tag model-server:$(IMAGE_TAG) $(IMAGE)
	docker push $(IMAGE)
	gcloud run jobs update $(GCP_JOB_NAME) --image=$(IMAGE) --region=$(GCP_REGION) --project=$(GCP_PROJECT_ID)

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache
