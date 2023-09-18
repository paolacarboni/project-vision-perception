PYTHON = python3
PIP = pip3

all: install run

install:
	$(PIP) install	-r	requirements.txt
	$(PIP) install --upgrade -r requirements.txt

datasets:
	$(PYTHON) srcs/main.py 3

train:
	$(PYTHON) srcs/main.py 2

run:
	$(PYTHON) srcs/main.py 1

clean:
	rm	-rf __pycache__ *.pyc
	rm	-rf */__pycache__ *.pyc
	find . -type d -name __pycache__ -exec rm -r {} \;