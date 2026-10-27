PYTHON ?= python3

.PHONY: install preprocess run test check

install:
	$(PYTHON) -m pip install -r requirements.txt

preprocess:
	$(PYTHON) preprocess.py

run:
	streamlit run app.py

test:
	$(PYTHON) -m pytest tests -q

check:
	$(PYTHON) -m py_compile app.py preprocess.py vibration_dashboard/*.py
	$(PYTHON) -m pytest tests -q
