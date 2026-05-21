import pandas as pd

path_str = r"/Users/kamilzawitaj/Documents/Studia/PW OKNO - AiR/Praca Mgr/code/leak_simulation/output_folder/pickle"

signal_leak_long = pd.read_pickle(path_str + '/signals.pkl')
scenario_metadata = pd.read_pickle(path_str + '/scenario_metadata.pkl')

'''print(signal_leak_long)
print(scenario_metadata.to_string())'''

signal_leak_wide = signal_leak_long.melt(
    id_vars=['T', 'Node'], 
    var_name='Scenario_Name', 
    value_name='Signal_Value'
)

signal_leak_wide_with_meta = pd.merge(
    signal_leak_wide, 
    scenario_metadata, 
    on='Scenario_Name', 
    how='left'
)

signal_leak_wide_with_meta = signal_leak_wide_with_meta[signal_leak_wide_with_meta['is_outlier'] == False]

signal_leak_wide_final = signal_leak_wide_with_meta.pivot_table(
    index=[
        'Scenario_Name', 
        'leak_diameter_parameter', 
        'time_of_failure_h', 
        'leak_location', 
        'is_outlier',
        'T'
    ],
    columns='Node',
    values='Signal_Value'
).reset_index()

signal_leak_wide_final.columns = [str(col) for col in signal_leak_wide_final.columns]

signal_leak_wide_final['Is_Leak'] = (signal_leak_wide_final['T'] > (signal_leak_wide_final['time_of_failure_h'] * 3600)).astype(int)

signal_leak_wide_final.columns = [str(col) for col in signal_leak_wide_final.columns]

print(signal_leak_wide_final)

#%%

path_csv = r"/Users/kamilzawitaj/Documents/Studia/PW OKNO - AiR/Praca Mgr/code/leak_simulation/output_folder/csv"
signal_leak_wide_final.to_csv(path_csv+'/signal_leak_wide_final.csv')
print(signal_leak_wide_final.head(10))

#%%


df_signals = signal_leak_wide_final

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, f1_score
import pandas as pd
import numpy as np

# ==========================================
# KROK 1: Podział na Cechy (X) i Cel (y) oraz Metadane
# ==========================================

# Tworzymy cel – to chcemy przewidzieć
y = df_signals['Is_Leak']

# Wybieramy tylko kolumny z czujnikami (nazwy węzłów)
# Odrzucamy wszystkie metadane i kolumnę celu
metadane_kolumny = ['Scenario_Name', 'leak_diameter_parameter', 'time_of_failure_h', 'leak_location', 'is_outlier', 'T', 'Is_Leak']
X = df_signals.drop(columns=metadane_kolumny, errors='ignore')

# Zachowujemy metadane dla późniejszej analizy (np. filtrowania w Streamlit)
metadata = df_signals[['Scenario_Name', 'leak_diameter_parameter', 'time_of_failure_h', 'leak_location', 'is_outlier', 'T']]

# ==========================================
# KROK 2: Podział na zbiór Treningowy i Testowy
# ==========================================
# WAŻNE: W wyciekach najlepiej dzielić po SCENARIUSZACH, a nie losowych wierszach, 
# żeby model nie widział fragmentów tego samego wycieku w teście.
unikalne_scenariusze = metadata['Scenario_Name'].unique()

scenariusze_train, scenariusze_test = train_test_split(
    unikalne_scenariusze, 
    test_size=0.3,          # 30% scenariuszy idzie do testów
    random_state=42        # żeby wyniki były powtarzalne
)

# Filtrujemy główną tabelę na podstawie wylosowanych scenariuszy
klucz_train = metadata['Scenario_Name'].isin(scenariusze_train)
klucz_test = metadata['Scenario_Name'].isin(scenariusze_test)

X_train, y_train = X[klucz_train], y[klucz_train]
X_test, y_test = X[klucz_test], y[klucz_test]
metadata_test = metadata[klucz_test].copy() # Zapamiętujemy metadane dla testu!

# ==========================================
# KROK 3: Definicja i Trening Modelu XGBoost
# ==========================================

# Ponieważ szumów (0) masz znacznie więcej niż punktów z wyciekiem (1), 
# obliczamy wagę, aby model nie ignorował wycieków:
liczba_szumow = np.sum(y_train == 0)
liczba_wyciekow = np.sum(y_train == 1)
waga_klas = liczba_szumow / liczba_wyciekow

# Inicjalizacja klasyfikatora
model_xgb = xgb.XGBClassifier(
    n_estimators=100,         # liczba drzew decyzyjnych
    max_depth=5,              # głębokość drzewa (zapobiega overfitingowi)
    learning_rate=0.1,        # szybkość uczenia
    scale_pos_weight=waga_klas,# ratuje nas przed niezbalansowanym zbiorem danych
    random_state=42,
    eval_metric='logloss'
)

# Trening!
model_xgb.fit(X_train, y_train)

# ==========================================
# KROK 4: Generowanie Predykcji (Prawdopodobieństw!)
# ==========================================

# Zamiast .predict() używamy .predict_proba(), żeby dostać szanse od 0.0 do 1.0
# [:, 1] oznacza, że bierzemy prawdopodobieństwo klasy 1 (czyli Wycieku)
prawdopodobienstwa = model_xgb.predict_proba(X_test)[:, 1]

# Dorzucamy te wyniki do naszych metadanych testowych
metadata_test['Leak_Probability'] = prawdopodobienstwa
metadata_test['True_Is_Leak'] = y_test

# ==========================================
# KROK 5: Wybór Progu i Ocena Modelu (Twój słynny Threshold!)
# ==========================================

# Ustalamy próg decyzyjny (np. 0.5 na start, ale docelowo będziesz nim sterował w Streamlit)
prog_decyzyjny = 0.5
metadata_test['Final_Prediction'] = (metadata_test['Leak_Probability'] >= prog_decyzyjny).astype(int)

# Wyświetlamy raport klasyfikacji
print("=== RAPORT KLASYFIKACJI ===")
print(classification_report(metadata_test['True_Is_Leak'], metadata_test['Final_Prediction']))