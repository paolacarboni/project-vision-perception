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

install: requirements.txt
	python3 -m venv $(VENV)
	pip install urllib3==1.26.6
	$(PIP) install -r requirements.txt

$(VENV)/bin/activate:
	python3 -m venv $(VENV)

clean:
	rm -rf __pycache__
	rm -rf $(VENV)
