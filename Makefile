PYTHON ?= python

.PHONY: data experiment report test

data:
	$(PYTHON) scripts/download_data.py

experiment: data
	$(PYTHON) scripts/run_experiment.py
	$(PYTHON) scripts/summarize.py

report:
	$(PYTHON) scripts/summarize.py

test:
	$(PYTHON) -m pytest -q

.PHONY: robustness
robustness: data
	$(PYTHON) scripts/run_robustness.py
