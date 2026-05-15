import pickle
from src.config import SIMULATION_CONFIG
import wntr
from src.alter_demand_model import get_alt_demand_wn
from src.get_3sigma_threshold import get_1sigma_threshold
from src.stochastic_simulation_signals import stochastic_simulation_signals_parallel
import pandas as pd
import re
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
import numpy as np

def generate_leak_signals():
    # signal_leak = stochastic_simulation_signals_parallel(leak_diameter_parameters, times_of_failure_h
    signal_leak = pd.read_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'signals.pkl')
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
    result = chama_outputs.loc[(chama_outputs['Budget'] == 4) & (chama_outputs['Formulation'] == 'CoverageFormulation')]
    sensors = result['Result'].item()['Sensors']
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

def get_precision_recall_df(target_leak_diameters, leak_signals, nodal_thresholds, sensors_picked): 

    thp_list = np.arange(.25, 5.25, 0.25).tolist()
    leak_signals = leak_signals.copy() 

    if target_leak_diameters is not None:
        if not isinstance(target_leak_diameters, list):
            target_leak_diameters = [target_leak_diameters]
            
        leak_signals = leak_signals[leak_signals['leak_diameter_parameter'].isin(target_leak_diameters)]
        
        if leak_signals.empty:
            return pd.DataFrame() # Zwracamy pusty df zamiast pustej listy

    leak_signals['T_seconds'] = leak_signals['time_of_failure_h'] * 3600

    # OPTYMALIZACJA 2: Przechodzimy na czyste tablice NumPy, eliminując narzut Pandasa
    signals_dict = {}
    for (scenario, node), group in leak_signals.groupby(['Scenario_Name', 'Node']):
        signals_dict[(scenario, str(node))] = (
            group['T'].values,                     # T_vals (NumPy array)
            group['Signal_Value'].values,          # Signal_vals (NumPy array)
            group['T_seconds'].iloc[0]             # tof_seconds (Wyliczane RAZ na stałe!)
        )

    precision_recall_data_list = []
    scenarios_list = leak_signals['Scenario_Name'].unique().tolist()
    
    # OPTYMALIZACJA 3: Zamiana iterrows() na itertuples() - wielokrotnie szybsze iterowanie
    sensors_picked_tuples = list(sensors_picked.itertuples(index=False))

    for thp in tqdm(thp_list, desc="Przetwarzanie threshold parameter", position=1, leave=False):
        for sensors_picked_row in tqdm(sensors_picked_tuples, desc="Przetwarzanie sensorów", position=2, leave=False):
            
            # W itertuples dobieramy się do kolumn po ich nazwie w obiekcie
            sensors_list_tuple = sensors_picked_row.Sensors_List_Int
            budget = sensors_picked_row.Budget
            formulation = sensors_picked_row.Formulation

            TP, FP, FN = 0, 0, 0

            for scenario in scenarios_list:
                system_alarmed_before_tof = False
                system_alarmed_after_tof = False
                
                for sensor in sensors_list_tuple:
                    node_id = str(sensor[0])

                    if (scenario, node_id) not in signals_dict:
                        continue
                    
                    # Pobieramy czyste tablice NumPy (błyskawiczny odczyt)
                    t_vals, sig_vals, tof_seconds = signals_dict[(scenario, node_id)]

                    threshold = thp * nodal_thresholds[node_id]

                    # Błyskawiczna wektoryzacja na tablicach NumPy
                    is_above_thresh = sig_vals >= threshold
                    is_before_tof = t_vals < tof_seconds
                    is_after_tof = t_vals >= tof_seconds

                    if (is_above_thresh & is_before_tof).any():
                        system_alarmed_before_tof = True
                    if (is_above_thresh & is_after_tof).any():
                        system_alarmed_after_tof = True

                if system_alarmed_before_tof:
                    FP += 1  

                if system_alarmed_after_tof:
                    TP += 1  
                else:
                    FN += 1  
            
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
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
                'leak_diameters': leak_info
            })

    precision_recall_data = pd.DataFrame(precision_recall_data_list)
    return precision_recall_data

def get_precision_recall_data():

    leak_signals = generate_leak_signals()
    
    # OPTYMALIZACJA 1: Wczytujemy pliki z dysku tylko raz!
    nodal_thresholds = pd.read_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'nodal_thresholds_std.pkl')
    sensors_picked = get_sensors_list_chama()
    # usunięto call do generate_bp_signals(), bo nie jest nigdzie używane.
    
    target_leak_diameters = leak_signals['leak_diameter_parameter'].unique().tolist()

    target_leak_diameters_list = []
    for target_leak_diameter in target_leak_diameters:
        target_leak_diameters_list.append([target_leak_diameter])
    target_leak_diameters_list.append(target_leak_diameters)

    # Przekazujemy wszystkie niezbędne stałe z zewnątrz
    def zadanie(target_diameter):
        return get_precision_recall_df(target_diameter, leak_signals, nodal_thresholds, sensors_picked)

    precision_recall_data_list = thread_map(
        zadanie,
        target_leak_diameters_list,
        max_workers=4,  # dostosuj do liczby wątków na PC
        desc="Przetwarzanie roznych leak diameter",
        position=0
    )

    precision_recall_data = pd.concat(precision_recall_data_list, ignore_index=True)
    return precision_recall_data


precision_recall_data = get_precision_recall_data()
print(precision_recall_data.to_string())