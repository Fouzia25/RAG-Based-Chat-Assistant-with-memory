.PHONY: help install up down logs ingest ingest-docker run ui test test-docker clean

PYTHON ?= python

help:
	@echo "make install        Install dependencies into the current Python env"
	@echo "make up             Start postgres, redis, api and ui with Docker Compose"
	@echo "make down           Stop all containers (data volumes are kept)"
	@echo "make logs           Follow API logs"
	@echo "make ingest         Index ./data locally (needs DB running)"
	@echo "make ingest-docker  Index the bundled sample docs inside the api container"
	@echo "make run            Run the FastAPI server locally with auto-reload"
	@echo "make ui             Run the Streamlit UI locally"
	@echo "make test           Run the test suite locally"
	@echo "make test-docker    Run the test suite inside Docker"

install:
	$(PYTHON) -m pip install -r requirements-dev.txt

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f api

ingest:
	$(PYTHON) -m app.ingestion.ingest --path ./data

ingest-docker:
	docker compose exec api python -m app.ingestion.ingest --path ./data/sample_docs

run:
	$(PYTHON) -m uvicorn app.main:app --reload --port 8000

ui:
	$(PYTHON) -m streamlit run ui/streamlit_app.py

test:
	$(PYTHON) -m pytest

test-docker:
	docker compose run --rm --no-deps -e OPENAI_API_KEY= api python -m pytest

clean:
	docker compose down -v
