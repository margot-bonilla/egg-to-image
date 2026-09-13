PYTHON := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: env test train zip-submission clean help

help:
	@echo "Available commands:"
	@echo "  make env            - Create Python 3.12 venv and install dependencies"
	@echo "  make test           - Run the offline smoke test suite"
	@echo "  make train          - Train model and export submission/weights.pt"
	@echo "  make zip-submission - Package submission folder for Codabench upload"
	@echo "  make clean          - Remove temporary caches"

env:
	/opt/homebrew/bin/python3.12 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

test:
	PYTHONPATH=. $(PYTHON) scripts/smoke_test.py

train:
	PYTHONPATH=. $(PYTHON) src/train.py

zip-submission:
	@mkdir -p dist
	cd submission && zip -r ../dist/submission.zip submission.py weights.pt
	@echo "Created dist/submission.zip ready for Codabench upload!"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf dist .pytest_cache

