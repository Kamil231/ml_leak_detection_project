import pickle
from src.config import SIMULATION_CONFIG
import wntr
from src.alter_demand_model import get_alt_demand_wn
from src.get_3sigma_threshold import get_1sigma_threshold
from src.stochastic_simulation_signals import stochastic_simulation_signals_parallel
import pandas as pd
import re
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map, process_map
from functools import partial
import numpy as np
import os

def generate_leak_signals():
    
    signal_leak = pd.read_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'signals_with_bp.pkl')
    signal_leak = signal_leak.drop(columns=['blueprint_scenario'])


    scenario_metadata = pd.read_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'scenario_metadata.pkl')

    signals_leak_melted = signal_leak.melt(
        id_vars=['T', 'Node'],  
        var_name='Scenario_Name',     
        value_name='Signal_Value'    
    )

    signals_long = pd.merge(
        signals_leak_melted,
        scenario_metadata, 
        on='Scenario_Name',
        how='left'
    )

    return signals_long

def generate_bp_signals(seed_offset = 0):

    wn_base = SIMULATION_CONFIG.create_network_base()
    sim_base = wntr.sim.WNTRSimulator(wn_base)
    results_base = sim_base.run_sim()

    wn = SIMULATION_CONFIG.create_network_real(seed_offset)
    sim_real = wntr.sim.WNTRSimulator(wn)
    results_real = sim_real.run_sim()

    residuals_matrix = results_base.node['pressure'] - results_real.node['pressure']                    
    residuals_stacked = residuals_matrix.stack()
    scenario_name = f'blueprint_scenario'
    residuals_stacked.name = scenario_name

    signal_final = residuals_stacked.reset_index()
    signal_final.rename(columns={'level_0': 'T', 'level_1': 'Node'}, inplace=True) 

    return signal_final

def get_sensors_list_chama():
    
    chama_outputs = pd.read_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'chama_outputs.pkl')
    sensors_picked_rows = []

    for row in chama_outputs.iterrows():
        sensors_picked_rows.append({'Budget': row[1]['Budget'], 'Formulation': row[1]['Formulation'], 'Sensors_List': row[1]['Result']['Sensors']})

    sensors_picked = pd.DataFrame(sensors_picked_rows)

    def parse_sensors_to_tuples(sensor_list):
        tuples_list = []
        for sensor in sensor_list:
            node_match = re.search(r'Node(\d+)', sensor)
            thp_match = re.search(r'thp(\d+)', sensor)
            
            if node_match and thp_match:
                node_id = int(node_match.group(1))
                thp = int(thp_match.group(1))
                tuples_list.append((node_id, thp))
                
        return tuples_list

    sensors_picked['Sensors_List_Int'] = sensors_picked['Sensors_List'].apply(parse_sensors_to_tuples)
    return sensors_picked

def get_precision_recall_leak_df(target_leak_diameters, leak_signals, nodal_thresholds, sensors_picked, precomputed_bp): 

    # thp_list = np.arange(.25, 5.25, 0.25).tolist()
    thp_list = np.arange(2, 5.25, 0.1).tolist()
    leak_signals = leak_signals.copy() 

    if target_leak_diameters is not None:
        if not isinstance(target_leak_diameters, list):
            target_leak_diameters = [target_leak_diameters]
            
        leak_signals = leak_signals[leak_signals['leak_diameter_parameter'].isin(target_leak_diameters)]
        
        if leak_signals.empty:
            return pd.DataFrame() 

    leak_signals['T_seconds'] = leak_signals['time_of_failure_h'] * 3600

    signals_dict = {}
    for (scenario, node), group in leak_signals.groupby(['Scenario_Name', 'Node']):
        signals_dict[(scenario, str(node))] = (
            group['T'].values,                     
            group['Signal_Value'].values,         
            group['T_seconds'].iloc[0]             
        )

    precision_recall_data_list = []
    scenarios_list = leak_signals['Scenario_Name'].unique().tolist()
    
    sensors_picked_tuples = list(sensors_picked.itertuples(index=False))

    for thp in tqdm(thp_list, desc="Przetwarzanie threshold parameter", position=1, leave=False):
        for sensors_picked_row in tqdm(sensors_picked_tuples, desc="Przetwarzanie sensorów", position=2, leave=False):
            
            sensors_list_tuple = sensors_picked_row.Sensors_List_Int
            budget = sensors_picked_row.Budget
            formulation = sensors_picked_row.Formulation

            TP, FP, TN, FN = 0, 0, 0, 0

            for scenario in tqdm(scenarios_list, desc="Przetwarzanie scenarios_list", position=3, leave=False):
                system_alarmed_before_tof = False
                system_alarmed_after_tof = False
                
                for sensor in sensors_list_tuple:
                    node_id = str(sensor[0])

                    if (scenario, node_id) not in signals_dict:
                        continue
                    
                    t_vals, sig_vals, tof_seconds = signals_dict[(scenario, node_id)]

                    #threshold = thp * nodal_thresholds[node_id]
                    m_node = nodal_thresholds.loc[node_id, 'mean']
                    s_node = nodal_thresholds.loc[node_id, 'std']

                    threshold = m_node + (thp * s_node)

                    is_above_thresh = sig_vals >= threshold
                    is_before_tof = t_vals < tof_seconds
                    is_after_tof = t_vals >= tof_seconds

                    if (is_above_thresh & is_before_tof).any():
                        system_alarmed_before_tof = True
                    if (is_above_thresh & is_after_tof).any():
                        system_alarmed_after_tof = True

                if system_alarmed_before_tof:
                    FP += 1  
                else:
                    TN += 1

                if system_alarmed_after_tof:
                    TP += 1  
                else:
                    FN += 1  

                FP_bp_base = 0
                TN_bp_base = 0
                
                for seed in range(len(precomputed_bp)):
                    if seed not in precomputed_bp: 
                        continue
                    
                    bp_node_dict = precomputed_bp[seed]
                    system_alarmed_bp = False
                    
                    for sensor in sensors_list_tuple:
                        node_id = str(sensor[0])
                        if node_id not in bp_node_dict:
                            continue
                            
                        sig_vals = bp_node_dict[node_id]
                        
                        #threshold = thp * nodal_thresholds[node_id]
                        m_node = nodal_thresholds.loc[node_id, 'mean']
                        s_node = nodal_thresholds.loc[node_id, 'std']

                        threshold = m_node + (thp * s_node)
                        
                        if (sig_vals >= threshold).any():
                            system_alarmed_bp = True
                            break 
                    
                    if system_alarmed_bp:
                        FP_bp_base += 1
                    else:
                        TN_bp_base += 1

                FP += FP_bp_base
                TN += TN_bp_base

            
            precision = TP / (TP + FP) if (TP + FP) > 0 else np.nan
            recall = TP / (TP + FN) if (TP + FN) > 0 else np.nan
            
            if pd.notna(precision) and pd.notna(recall) and (precision + recall) > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score = np.nan
            
            if target_leak_diameters is None:
                leak_info = 'None'
            elif len(target_leak_diameters) > 1:
                leak_info = 'All'
            else:
                leak_info = ", ".join(map(str, target_leak_diameters))
                
            precision_recall_data_list.append({
                'budget': budget, 
                'formulation': formulation,
                'thp': thp,
                'precision': precision, 
                'recall': recall,
                'f1_score': f1_score,
                'leak_diameters': leak_info,
                'TP': TP,
                'FP': FP,
                'TN': TN,
                'FN': FN
            })

    precision_recall_data = pd.DataFrame(precision_recall_data_list)
    return precision_recall_data

def _process_target_diameter(target_diameter, leak_signals, nodal_thresholds, sensors_picked, precomputed_bp):
    return get_precision_recall_leak_df(target_diameter, leak_signals, nodal_thresholds, sensors_picked, precomputed_bp)

def get_precision_recall_data():

    leak_signals = generate_leak_signals()
    
    nodal_thresholds = pd.read_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'nodal_thresholds_std.pkl')
    sensors_picked = get_sensors_list_chama()

    precomputed_bp = precompute_bp_signals_dict(max_seed=SIMULATION_CONFIG.dataset_parameters.number_of_BP_scenarios)
    
    target_leak_diameters = leak_signals['leak_diameter_parameter'].unique().tolist()

    target_leak_diameters_list = []
    for target_leak_diameter in target_leak_diameters:
        target_leak_diameters_list.append([target_leak_diameter])
    target_leak_diameters_list.append(target_leak_diameters)

    worker_func = partial(
        _process_target_diameter,
        leak_signals=leak_signals,
        nodal_thresholds=nodal_thresholds,
        sensors_picked=sensors_picked,
        precomputed_bp=precomputed_bp
    )

    precision_recall_data_list = process_map(
        worker_func,
        target_leak_diameters_list,
        max_workers=os.cpu_count(),  
        desc="Przetwarzanie roznych leak diameter"
    )

    precision_recall_data = pd.concat(precision_recall_data_list, ignore_index=True)
    return precision_recall_data

def precompute_bp_signals_dict(max_seed=120):
    bp_dict = {}
    for seed in tqdm(range(max_seed), desc="WNTR Base Simulations"):
        df_bp = generate_bp_signals(seed)
        node_dict = {}
        for node, group in df_bp.groupby('Node'):
            node_dict[str(node)] = group['blueprint_scenario'].values
            
        bp_dict[seed] = node_dict
        
    return bp_dict

def get_sensors_list_chama_seperate():

    chama_outputs = pd.read_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'chama_outputs_seperate.pkl')
    sensors_picked_rows = []

    for row in chama_outputs.iterrows():
        sensors_picked_rows.append({
            'Leak_Diameter': row[1]['Leak_Diameter'],
            'Budget': row[1]['Budget'], 
            'Formulation': row[1]['Formulation'], 
            'Sensors_List': row[1]['Result']['Sensors']
        })

    sensors_picked = pd.DataFrame(sensors_picked_rows)

    def parse_sensors_to_tuples(sensor_list):
        tuples_list = []
        for sensor in sensor_list:
            node_match = re.search(r'Node(\d+)', sensor)
            thp_match = re.search(r'thp(\d+)', sensor)
            
            if node_match and thp_match:
                node_id = int(node_match.group(1))
                thp = int(thp_match.group(1))
                tuples_list.append((node_id, thp))
                
        return tuples_list

    sensors_picked['Sensors_List_Int'] = sensors_picked['Sensors_List'].apply(parse_sensors_to_tuples)
    return sensors_picked

def get_precision_recall_leak_df_seperate(target_leak_diameter, leak_signals, nodal_thresholds, sensors_picked, precomputed_bp): 

    thp_list = np.arange(2, 5.25, 0.1).tolist()
    
    # leak_signals_filtered = leak_signals[leak_signals['leak_diameter_parameter'] == target_leak_diameter].copy()
    leak_signals_filtered = leak_signals[
        (leak_signals['leak_diameter_parameter'] == target_leak_diameter) | 
        (leak_signals['leak_diameter_parameter'].isna())
    ].copy()
    
    if leak_signals_filtered.empty:
        return pd.DataFrame() 

    leak_signals_filtered['T_seconds'] = leak_signals_filtered['time_of_failure_h'] * 3600

    signals_dict = {}
    for (scenario, node), group in leak_signals_filtered.groupby(['Scenario_Name', 'Node']):
        signals_dict[(scenario, str(node))] = (
            group['T'].values,                     
            group['Signal_Value'].values,         
            group['T_seconds'].iloc[0]             
        )

    precision_recall_data_list = []
    scenarios_list = leak_signals_filtered['Scenario_Name'].unique().tolist()

    sensors_picked_filtered = sensors_picked[sensors_picked['Leak_Diameter'] == target_leak_diameter]
    sensors_picked_tuples = list(sensors_picked_filtered.itertuples(index=False))

    for thp in tqdm(thp_list, desc=f"THP dla wycieku {target_leak_diameter}", position=1, leave=False):
        for sensors_picked_row in tqdm(sensors_picked_tuples, desc="Budżety & Formulation", position=2, leave=False):
            
            sensors_list_tuple = sensors_picked_row.Sensors_List_Int
            budget = sensors_picked_row.Budget
            formulation = sensors_picked_row.Formulation
            ldp_val = sensors_picked_row.Leak_Diameter

            TP, FP, TN, FN = 0, 0, 0, 0

            for scenario in scenarios_list:
                system_alarmed_before_tof = False
                system_alarmed_after_tof = False
                
                for sensor in sensors_list_tuple:
                    node_id = str(sensor[0])

                    if (scenario, node_id) not in signals_dict:
                        continue
                    
                    t_vals, sig_vals, tof_seconds = signals_dict[(scenario, node_id)]

                    m_node = nodal_thresholds.loc[node_id, 'mean']
                    s_node = nodal_thresholds.loc[node_id, 'std']
                    threshold = m_node + (thp * s_node)

                    is_above_thresh = sig_vals >= threshold
                    is_before_tof = t_vals < tof_seconds
                    is_after_tof = t_vals >= tof_seconds

                    if (is_above_thresh & is_before_tof).any():
                        system_alarmed_before_tof = True
                    if (is_above_thresh & is_after_tof).any():
                        system_alarmed_after_tof = True

                if system_alarmed_before_tof:
                    FP += 1  
                else:
                    TN += 1

                if system_alarmed_after_tof:
                    TP += 1  
                else:
                    FN += 1  

            FP_bp_base = 0
            TN_bp_base = 0
            
            for seed in range(len(precomputed_bp)):
                if seed not in precomputed_bp: 
                    continue
                
                bp_node_dict = precomputed_bp[seed]
                system_alarmed_bp = False
                
                for sensor in sensors_list_tuple:
                    node_id = str(sensor[0])
                    if node_id not in bp_node_dict:
                        continue
                        
                    sig_vals = bp_node_dict[node_id]
                    
                    m_node = nodal_thresholds.loc[node_id, 'mean']
                    s_node = nodal_thresholds.loc[node_id, 'std']
                    threshold = m_node + (thp * s_node)
                    
                    if (sig_vals >= threshold).any():
                        system_alarmed_bp = True
                        break 
                
                if system_alarmed_bp:
                    FP_bp_base += 1
                else:
                    TN_bp_base += 1

            FP += FP_bp_base
            TN += TN_bp_base

            precision = TP / (TP + FP) if (TP + FP) > 0 else np.nan
            recall = TP / (TP + FN) if (TP + FN) > 0 else np.nan
            
            if pd.notna(precision) and pd.notna(recall) and (precision + recall) > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score = np.nan
                
            precision_recall_data_list.append({
                'budget': budget, 
                'formulation': formulation,
                'thp': thp,
                'precision': precision, 
                'recall': recall,
                'f1_score': f1_score,
                'leak_diameters': str(ldp_val), 
                'TP': TP,
                'FP': FP,
                'TN': TN,
                'FN': FN
            })

    return pd.DataFrame(precision_recall_data_list)

def _process_target_diameter_seperate(target_diameter, leak_signals, nodal_thresholds, sensors_picked, precomputed_bp):
    return get_precision_recall_leak_df_seperate(target_diameter, leak_signals, nodal_thresholds, sensors_picked, precomputed_bp)

def get_precision_recall_data_seperate():

    leak_signals = generate_leak_signals()
    nodal_thresholds = pd.read_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'nodal_thresholds_std.pkl')
    
    sensors_picked = get_sensors_list_chama_seperate()

    precomputed_bp = precompute_bp_signals_dict(max_seed=SIMULATION_CONFIG.dataset_parameters.number_of_BP_scenarios)
    
    target_leak_diameters = leak_signals['leak_diameter_parameter'].unique().tolist()

    worker_func = partial(
        _process_target_diameter_seperate,
        leak_signals=leak_signals,
        nodal_thresholds=nodal_thresholds,
        sensors_picked=sensors_picked,
        precomputed_bp=precomputed_bp
    )

    precision_recall_data_list = process_map(
        worker_func,
        target_leak_diameters, 
        max_workers=os.cpu_count(),  
        desc="Ewaluacja osobnych średnic wycieków"
    )

    precision_recall_data_seperate = pd.concat(precision_recall_data_list, ignore_index=True)
    return precision_recall_data_seperate
    

# precision_recall_data = get_precision_recall_data()
# precision_recall_data.to_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'precision_recall_data_chama.pkl')
# precision_recall_data.to_csv(SIMULATION_CONFIG.output_folder / 'csv' / 'precision_recall_data_chama.csv')

# precision_recall_data = get_precision_recall_data()
# print(precision_recall_data.to_string())
# print(type(precision_recall_data))

# bp_signal = generate_bp_signals()
# print(bp_signal)

# leak_signals = generate_leak_signals()
# print(leak_signals.head(10).to_string())

