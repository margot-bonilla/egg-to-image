PYTHON := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: env env-create env-install competition-repo test train train-debug zip-submission test-submission clean help

help:
	@echo "Available commands:"
	@echo "  make env              - Create the venv, install dependencies, clone the competition repo"
	@echo "  make test             - Run the offline smoke test suite"
	@echo "  make train            - Train model and export submission/weights.pt"
	@echo "  make train-debug      - Train on synthetic warm-up-shaped data via Benchopt (pipeline check only)"
	@echo "  make zip-submission   - Package submission folder for Codabench upload"
	@echo "  make test-submission  - Zip, then run it read-only like Codabench does"
	@echo "  make clean            - Remove temporary caches"

env: env-create env-install competition-repo

env-create:
	python3 -m venv .venv

env-install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# Official competition repo: provides benchmark_utils, the scoring objective and
# the datasets used by Codabench. Kept out of git (see .gitignore).
COMPET_REPO := external/2026-competition

competition-repo:
	@test -d $(COMPET_REPO) || git clone --depth 1 https://github.com/neural-interfaces26/2026-competition $(COMPET_REPO)

test: competition-repo
	PYTHONPATH=. $(PYTHON) scripts/smoke_test.py

train:
	PYTHONPATH=. $(PYTHON) src/train.py

# Synthetic data with the warm-up shape (63 channels x 120 samples -> 1536).
WARMUP_SIM := Simulated[n_chans=63,n_times=120,n_images=50,n_outputs=1536,n_train=200,n_test=400,sfreq=120]
WARMUP_SIM_TRAIN := Simulated[n_chans=63,n_times=120,n_images=50,n_outputs=1536,n_train=2000,n_test=400,sfreq=120]

# Reproduces the current debug submission/weights.pt. Scores are meaningless on
# real data; it only proves the train -> export -> reload path works.
train-debug: competition-repo
	cd $(COMPET_REPO) && PATH="$(CURDIR)/.venv/bin:$$PATH" \
		benchopt run tracks/image_decoding -o "Image-decoding[training=True]" \
		-d "$(WARMUP_SIM_TRAIN)" -s "$(CURDIR)/submission/submission.py" --no-plot --no-cache
	cp $(COMPET_REPO)/tracks/image_decoding/outputs/EEGToImageBaseline/weights.pt submission/weights.pt

zip-submission:
	@mkdir -p dist
	rm -f dist/submission.zip
	zip -j dist/submission.zip submission/submission.py submission/weights.pt
	@echo "Created dist/submission.zip ready for Codabench upload!"

# Mimic Codabench: unzip into a read-only folder and run inference-only.
test-submission: competition-repo zip-submission
	rm -rf dist/unzipped && mkdir -p dist/unzipped
	unzip -q dist/submission.zip -d dist/unzipped
	chmod -R a-w dist/unzipped
	cd $(COMPET_REPO) && PATH="$(CURDIR)/.venv/bin:$$PATH" \
		COMPET_SUBMISSION_DIR="$(CURDIR)/dist/unzipped" \
		benchopt run tracks/image_decoding -d "$(WARMUP_SIM)" \
		-s "$(CURDIR)/dist/unzipped/submission.py" --no-plot --no-cache
	chmod -R u+w dist/unzipped

clean:
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
	chmod -R u+w dist 2>/dev/null || true
	rm -rf dist .pytest_cache
