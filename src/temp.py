from src.config import SIMULATION_CONFIG
import wntr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import pickle
import re
import numpy as np

pickle_path = '/Users/kamilzawitaj/Documents/Studia/PW OKNO - AiR/Praca Mgr/code/leak_simulation/output_folder/pickle/'

def load_data_signals():  
    df = pd.read_pickle(pickle_path + 'signals.pkl')   
    return df

def return_outlier_scenario(df_signals):

    df_signals = load_data_signals()

    df_signals['T'] = df_signals['T'] / 3600

    limit_time = 16
    df_subset = df_signals[df_signals['T'] <= limit_time]

    scenario_cols = [
        c for c in df_signals.columns 
        if (match := re.search(r'tofh(\d+)', c)) and int(match.group(1)) > 24
    ]

    errors_all = df_subset[scenario_cols].sub(df_subset['blueprint_scenario'], axis=0)

    global_mean = errors_all.values.mean()
    global_std = errors_all.values.std()

    upper_limit = global_mean + 3 * global_std
    lower_limit = global_mean - 3 * global_std

    is_outlier = ((errors_all > upper_limit) | (errors_all < lower_limit)).any()
    outlier_names = is_outlier[is_outlier == True].index.tolist()

    if not outlier_names:
        print("Nie znaleziono outlierów.")
        return None
    else:
        print("Wykryte outliery:")
        print('\n'.join(outlier_names))
        return outlier_names