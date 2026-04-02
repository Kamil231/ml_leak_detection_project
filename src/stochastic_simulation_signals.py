import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='wntr')
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="chama")
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
import numpy as np
import pickle
import wntr
import pandas as pd
from src.config import SIMULATION_CONFIG
import os
import copy
from tqdm import tqdm
from itertools import product
from joblib import Parallel, delayed
import itertools
from collections import defaultdict
import copy
from src.alter_demand_model import get_alt_demand_wn

def stochastic_simulation_signals(leak_diameter_parameters, times_of_failure_h):
    
    wn = SIMULATION_CONFIG.create_newtork()

    sim = wntr.sim.WNTRSimulator(wn)
    #ref_sim = sim.run_sim()
    pipe_names = wn.pipe_name_list
    dfs_list = []

    scenario_metadata = []

    with tqdm(total=len(leak_diameter_parameters)*len(times_of_failure_h)*len(pipe_names), desc="Leak calculations") as pbar:
        for leak_diameter_parameter in leak_diameter_parameters:
            for time_of_failure_h in times_of_failure_h:
                for pipe_to_fail in pipe_names:
                
                    wn_temp = SIMULATION_CONFIG.create_newtork()
                    
                    start_time_s = time_of_failure_h * 3600
                
                    if SIMULATION_CONFIG.time.leak_duration != 0:
                        end_time_s = (time_of_failure_h + SIMULATION_CONFIG.time.leak_duration) * 3600
                    else:
                        end_time_s = SIMULATION_CONFIG.time.duration_s
                
                    pipe = wn_temp.get_link(pipe_to_fail)
                    leak_diameter = pipe.diameter * leak_diameter_parameter
                    leak_area = np.pi * (leak_diameter / 2)**2
                    
                    node_name = pipe_to_fail + '_leak_node'
                    wn_temp = wntr.morph.split_pipe(wn_temp, pipe_to_fail, pipe_to_fail + '_A', node_name)
                    
                    leak_node = wn_temp.get_node(node_name) #new

                    wn_ref_local = copy.deepcopy(wn_temp)

                    wn_ref_local = get_alt_demand_wn(wn_ref_local) 

                    sim_ref = wntr.sim.WNTRSimulator(wn_ref_local)
                    results_ref = sim_ref.run_sim()

                    leak_node.add_leak(wn_temp, area=leak_area, start_time=start_time_s, end_time=end_time_s)
                    sim_leak = wntr.sim.WNTRSimulator(wn_temp)
                    results_leak = sim_leak.run_sim()

                    residuals_matrix = results_ref.node['pressure'] - results_leak.node['press ure']
                    
                    residuals_stacked = residuals_matrix.stack()
                    scenario_name = f'ldp{leak_diameter_parameter}_tofh{time_of_failure_h}_pl{pipe_to_fail}_scenario'
                    residuals_stacked.name = scenario_name
                    dfs_list.append(residuals_stacked)     

                    scenario_metadata.append({
                        'Scenario_Name': scenario_name,
                        'leak_diameter_parameter': leak_diameter_parameter,
                        'time_of_failure_h': time_of_failure_h,
                        'leak_location': pipe_to_fail
                    })

                    pbar.update(1)

                        
                            
    signal_final = pd.concat(dfs_list, axis=1)
    
    signal_final = signal_final.reset_index()
    signal_final.rename(columns={'level_0': 'T', 'level_1': 'Node'}, inplace=True) 
    
    # signal_final['leak_diameter_parameter'] = leak_diameter_parameter
    # signal_final['time_of_failure_h'] = time_of_failure_h

    metadata_df = pd.DataFrame(scenario_metadata)
    metadata_df.to_csv(SIMULATION_CONFIG.output_folder + '/csv/scenario_metadata.csv')
    metadata_df.to_pickle(SIMULATION_CONFIG.output_folder + "/pickle/scenario_metadata.pkl")

    
    return signal_final
 
def run_single_leak(ldp, tofh, pipe_to_fail):

    wn_temp = SIMULATION_CONFIG.create_newtork()
    
    start_time_s = tofh * 3600
    if SIMULATION_CONFIG.time.leak_duration != 0:
        end_time_s = (tofh + SIMULATION_CONFIG.time.leak_duration) * 3600
    else:
        end_time_s = SIMULATION_CONFIG.time.duration_s

    wn_temp.options.hydraulic.demand_model = 'PDD' 

    # Konfiguracja wycieku
    pipe = wn_temp.get_link(pipe_to_fail)
    leak_diameter = pipe.diameter * ldp
    leak_area = np.pi * (leak_diameter / 2)**2
    
    node_name = f"{pipe_to_fail}_leak_node"
    wn_temp = wntr.morph.split_pipe(wn_temp, pipe_to_fail, pipe_to_fail + '_A', node_name)
    
    wn_ref = copy.deepcopy(wn_temp)
    sim_ref = wntr.sim.WNTRSimulator(wn_ref)
    results_ref = sim_ref.run_sim()

    wn_temp = get_alt_demand_wn(wn_temp) 

    leak_node = wn_temp.get_node(node_name)
    leak_node.add_leak(wn_temp, 
                       area=leak_area,
                       start_time=start_time_s,
                       end_time=end_time_s)
    
    
    # sim = wntr.sim.WNTRSimulator(wn_temp)
    sim_leak = wntr.sim.WNTRSimulator(wn_temp)
    results_leak = sim_leak.run_sim()

    residuals_matrix = results_ref.node['pressure'] - results_leak.node['pressure']
        
    # Konwersja do "długiego formatu"
    res_df = residuals_matrix.stack().reset_index()
    res_df.columns = ['T', 'Node', 'Pressure_Residual']
    
    # Dodanie metadanych
    res_df['leak_diameter_parameter'] = ldp
    res_df['time_of_failure_h'] = tofh
    res_df['leak_location'] = pipe_to_fail
    
    return res_df

def stochastic_simulation_signals_parallel(leak_diameter_parameters, times_of_failure_h, n_jobs=-1):
    # 1. Symulacja referencyjna
    wn = SIMULATION_CONFIG.create_newtork()
    sim = wntr.sim.WNTRSimulator(wn)
    ref_sim = sim.run_sim()
    ref_pressure_matrix = ref_sim.node['pressure']
    pipe_names = wn.pipe_name_list

    # 2. Przygotowanie zadań
    tasks = list(itertools.product(leak_diameter_parameters, times_of_failure_h, pipe_names))

    print('tasks[1]: ', tasks[1])
    print('type(tasks[1]): ', type(tasks[1]))
    t_dict = defaultdict(list)

    for index, task in enumerate(tasks):
        t_dict[task].append(index)

    # Wyświetl tylko te, które mają więcej niż jeden indeks
    for task, indices in t_dict.items():
        if len(indices) > 1:
            print(f"Tupla {task} występuje na indeksach: {indices}")
    
    # 3. Obliczenia równoległe
    results_list = Parallel(n_jobs=n_jobs)(
        # delayed(run_single_leak)(ldp, tofh, pipe, ref_pressure_matrix)
        delayed(run_single_leak)(ldp, tofh, pipe) 
        for ldp, tofh, pipe in tqdm(tasks, desc="Parallel Leak Calculations")
    )

    # 4. Łączenie wyników w jeden "długi" DataFrame
    results_list = [r for r in results_list if r is not None]
    
    # Zabezpieczenie na wypadek gdyby wszystkie symulacje się nie powiodły
    if not results_list:
        print("Błąd: Brak wyników symulacji.")
        return pd.DataFrame(), pd.DataFrame()

    df_long = pd.concat(results_list, axis=0, ignore_index=True)

    # --- TWORZENIE NAZWY SCENARIUSZA ---
    df_long['Scenario_Name'] = (
        'ldp' + df_long['leak_diameter_parameter'].astype(str) + 
        '_tofh' + df_long['time_of_failure_h'].astype(str) + 
        '_pl' + df_long['leak_location'] + '_scenario'
    )

    # --- KROK 1: EKSTRAKCJA METADANYCH (NOWOŚĆ) ---
    # Wybieramy tylko kolumny opisujące scenariusz i usuwamy duplikaty (bo te dane powtarzają się dla każdego węzła i kroku czasu)
    metadata_cols = ['Scenario_Name', 'leak_diameter_parameter', 'time_of_failure_h', 'leak_location']
    scenario_metadata = df_long[metadata_cols].drop_duplicates(subset='Scenario_Name').reset_index(drop=True)

    # --- KROK 2: PIVOT DO FORMATU SZEROKIEGO DLA CHAMY ---
    signal_final = df_long.pivot(
        index=['T', 'Node'], 
        columns='Scenario_Name', 
        values='Pressure_Residual'
    ).reset_index()

    #print(signal_final.columns.tolist())

    # Usuwamy błędne przypisanie pojedynczych wartości na końcu tabeli, które było w starym kodzie
    # signal_final['leak_diameter_parameter'] = ... (USUNIĘTE)

    # Opcjonalnie: Zapis metadanych wewnątrz funkcji (tak jak w Twoim kodzie sekwencyjnym)
    # Choć lepiej robić to na zewnątrz funkcji.
    metadata_df = pd.DataFrame(scenario_metadata)
    metadata_df.to_csv(SIMULATION_CONFIG.output_folder / 'csv' / 'scenario_metadata.csv')
    metadata_df.to_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'scenario_metadata.pkl')

    signal_final = signal_final[signal_final['Node'].isin(wn.node_name_list)]

    return signal_final
