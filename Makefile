SHELL := /bin/bash

PYTHON = python
PIP = $(PYTHON) -m pip
MODULE = src.main
STREAMLIT = streamlit
DASHBOARD = src/display_plots_st.py

UNAME_S := $(shell uname -s)

.PHONY: help run install display

all: install run display

help:
	@echo "Dostępne komendy:"
	@echo "  make run      - Uruchamia obliczenia (python -m src.main)"
	@echo "  make install  - Instaluje biblioteki"
	@echo "  make display  - Uruchamia Streamlit z wykresami"

run:
	$(PYTHON) -m $(MODULE)

install:
	@echo "1. Sprawdzanie i instalacja środowiska Python"
	@if ! command -v $(PYTHON) &> /dev/null; then \
		echo "Brak Pythona w środowisku, instaluję Pythona i pip"; \
		conda install -y python pip; \
	else \
		echo "Python jest już zainstalowany."; \
	fi

	@echo "2. Instalacja pakietów (requirements.txt)"
	$(PIP) install -r requirements.txt
	
	@echo "3. Instalacja pakietów systemowych (libomp, glpk"
ifeq ($(UNAME_S),Darwin)
	@if ! command -v brew &> /dev/null; then \
		echo "Brak Homebrew"; \
		/bin/bash -c "$$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"; \
		eval "$$(/opt/homebrew/bin/brew shellenv)"; \
	else \
		echo "Homebrew jest już zainstalowany."; \
	fi
	brew install libomp glpk
else ifeq ($(UNAME_S),Linux)
	sudo apt-get update && sudo apt-get install -y libomp-dev glpk-utils
endif
	@echo "nstalacja zakończona"

display:
	PYTHONPATH=. $(STREAMLIT) run $(DASHBOARD)