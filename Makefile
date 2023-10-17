VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

datasets: $(VENV)/bin/activate
	$(PYTHON) srcs/main.py datasets

train: $(VENV)/bin/activate
	$(PYTHON) srcs/main.py train

run: $(VENV)/bin/activate
	$(PYTHON) srcs/main.py exec

test: $(VENV)/bin/activate
	$(PYTHON) srcs/main.py test

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

clean:
	rm -rf __pycache__
	rm -rf $(VENV)
