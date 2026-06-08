# Opracowanie oraz implementacja systemu do detekcji wycieków w sieciach wodociągowych przy wykorzystaniu metod sztucznej inteligencji.

Repozytorium zawiera kod źródłowy stworzony na potrzeby pracy magisterskiej/inżynierskiej pt. *"Opracowanie oraz implementacja systemu do detekcji wycieków w sieciach wodociągowych przy wykorzystaniu metod sztucznej inteligencji."*.

Projekt integruje modelowanie hydrauliczne sieci wodociągowych (oparte na silniku EPANET), stochastyczne generowanie profili zużycia wody, optymalizację rozmieszczenia czujników oraz i algorytmy sztucznej inteligencji w celu wczesnego wykrywania i lokalizacji awarii.

## Struktura repozytorium

* `data/` - Pliki wejściowe modeli sieci wodociągowych (`Net3.inp`).
* `src/` - Główny kod źródłowy projektu:
  * `generate_ml_dataset.py`, `stochastic_simulation_signals.py` - skrypty do generowania danych.
  * `run_chama_analysis.py`, `sensors_CoverageFormulation.py`, `sensors_ImpactFormulation.py` - skrypty do optymalizacji rozmieszczenia czujnikow wykorzystujace biblioteke chamaa (benchmark).
  * `XGBoost_analysis.py`, `LightGBM_analysis.py`, `pytorch_nn.py` - skrypty trenujące i testujące(ROC, precision recall, confusion matrix, F1-score) modele ML.
  * `display_*.py` - wizualizacyja wyników w frameworku Streamlit
* `output_folder/` - Folder docelowy na wygenerowane zestawy danych i  symulacje (Ignorowany w git) .
* `Makefile` 
* `requirements.txt` - Lista bibliotek

Projekt wymaga języka Python

Dostępne komendy make:
* make run      - Uruchamia obliczenia (python -m src.main)
* make install  - Instaluje biblioteki
* make display  - Uruchamia Streamlit z wykresami
* make setup_data  - Pobiera dane z przeliczonego programu
