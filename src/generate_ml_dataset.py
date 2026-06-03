import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from src.config import SIMULATION_CONFIG
import wntr
from tqdm import tqdm
from sklearn.metrics import roc_curve
import pickle
from pprint import pprint
from joblib import Parallel, delayed

def generate_bp_signals(seed_offset = 0):
    
    wn_base = SIMULATION_CONFIG.create_network_base()
    sim_base = wntr.sim.WNTRSimulator(wn_base)
    results_base = sim_base.run_sim()

    wn = SIMULATION_CONFIG.create_network_real(seed_offset)
    sim_real = wntr.sim.WNTRSimulator(wn)
    results_real = sim_real.run_sim()

    residuals_matrix = results_base.node['pressure'] - results_real.node['pressure']                    
    residuals_stacked = residuals_matrix.stack()
    scenario_name = f'blueprint_scenario_{seed_offset}'
    residuals_stacked.name = scenario_name

    signal_final = residuals_stacked.reset_index()
    signal_final.rename(columns={'level_0': 'T', 'level_1': 'Node'}, inplace=True) 

    signal_final = signal_final.pivot_table(
        index='T',
        columns='Node',
        values=scenario_name
    )

    signal_final.columns.name = None
    signal_final = signal_final.reset_index()
    signal_final['Scenario_Name'] = scenario_name

    old_cols = list(signal_final.columns)
    old_cols.remove('Scenario_Name')
    old_cols.remove('T')
    new_order = ['Scenario_Name', 'T'] + old_cols
    signal_final = signal_final[new_order]

    return signal_final

def get_signals_df():

    scenario_metadata = pd.read_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'scenario_metadata.pkl')

    signal_leak_long = pd.read_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'signals_with_bp.pkl')

    signal_leak_long = signal_leak_long.drop(columns=['blueprint_scenario'])

    # max_seed = 2000
    max_seed = SIMULATION_CONFIG.dataset_parameters.number_of_BP_scenarios

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

    signal_leak_wide_with_meta['leak_diameter_parameter'] = signal_leak_wide_with_meta['leak_diameter_parameter'].round(4)

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

    results_list = Parallel(n_jobs=-1)(
        delayed(generate_bp_signals)(seed) 
        for seed in tqdm(range(max_seed), desc="Parallel WNTR Base Simulations")
    )

    df_temp = pd.concat(results_list, axis=0, ignore_index=True)

    df_temp['Is_Leak'] = 0

    df_final = pd.concat([signal_leak_wide_final, df_temp], axis=0, ignore_index=True)

    # df_final.to_csv(SIMULATION_CONFIG.output_folder / 'csv' / 'signals_ml_dataset.csv')
    df_final.to_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'signals_ml_dataset.pkl')


    return df_final