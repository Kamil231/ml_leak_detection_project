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
	@echo "  make run      - Uruchamia główny skrypt (python -m src.main)"
	@echo "  make install  - Instaluje zależności z requirements.txt"
	@echo "  make display  - Wyświetla wyniki"

run:
	$(PYTHON) -m $(MODULE)

install:
	$(PIP) install -r requirements.txt
ifeq ($(UNAME_S),Darwin)
	@echo "Wnstaluje libomp dla macos"
	brew install libomp glpk
else ifeq ($(UNAME_S),Linux)
	@echo "Instaluje libomp lunux"
	sudo apt-get update && sudo apt-get install -y libomp-dev glpk-utils
endif

display:
	PYTHONPATH=. $(STREAMLIT) run $(DASHBOARD)

