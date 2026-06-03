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
import re
from pprint import pprint

def get_blueprint_signals():

    wn_base = SIMULATION_CONFIG.create_network_base()
    sim_base = wntr.sim.WNTRSimulator(wn_base)
    results_base = sim_base.run_sim()

    wn = SIMULATION_CONFIG.create_network_real()
    sim_real = wntr.sim.WNTRSimulator(wn)
    results_real = sim_real.run_sim()

    residuals_matrix = results_base.node['pressure'] - results_real.node['pressure']                    
    residuals_stacked = residuals_matrix.stack()
    scenario_name = f'blueprint_scenario'
    residuals_stacked.name = scenario_name

    signal_final = residuals_stacked.reset_index()
    signal_final.rename(columns={'level_0': 'T', 'level_1': 'Node'}, inplace=True) 

    return signal_final

def return_outlier_scenario(df_signals, tof_list):

    max_tof = max(tof_list)
    max_tof = max_tof - 1

    if max_tof == 0:
        return None

    df_T_subset = df_signals[df_signals['T'] <= max_tof * 3600]
    
    scenarios = [
        c for c in df_T_subset.columns 
        if (match := re.search(r'tofh(\d+)', c)) and int(match.group(1)) >= max_tof
    ]
    
    
    ldps = sorted(list(set(s.split('_')[0] for s in scenarios)))
    tofhs = sorted(list(set(s.split('_')[1] for s in scenarios)))
    nodes = df_T_subset['Node'].unique()
    
    grouped_data = defaultdict(lambda: defaultdict(list))

    for s in scenarios:
        parts = s.split('_')
        l_val = parts[0]
        t_val = parts[1]
        grouped_data[l_val][t_val].append(s)
    
    df_wide = df_T_subset.pivot(index='T', columns='Node', values=scenarios)
    
    outlier_scenarios = []
    
    for ldp in ldps:
        for tofh in tofhs:
            for node in nodes:
                
                filtered_scenarios = grouped_data[ldp][tofh]
                
                idx = pd.IndexSlice

                df_subset = df_wide.loc[:, idx[filtered_scenarios, node]]
                
                mean_course = df_subset.mean(axis=1)
                
                errors = df_subset.sub(mean_course, axis=0)
                
                global_mean = errors.values.mean().mean()
                global_std = errors.values.std().mean()
                
                upper_limit = global_mean + 3 * global_std
                lower_limit = global_mean - 3 * global_std
                
                is_outlier = (errors > upper_limit) | (errors < lower_limit)
                outlier_mask = is_outlier.any(axis=0)
                outlier_scenarios_and_nodes = outlier_mask[outlier_mask == True].index.tolist()
                outlier_scenarios.append(outlier_scenarios_and_nodes)
    
    unique_outlier_scenarios = list(set([sublist[0][0] for sublist in outlier_scenarios if sublist]))

    print('outlier_scenarios: ', unique_outlier_scenarios)
    
    return unique_outlier_scenarios

def stochastic_simulation_signals(leak_diameter_parameters, times_of_failure_h):
    
    wn = SIMULATION_CONFIG.create_network_real()
    sim = wntr.sim.WNTRSimulator(wn)
    pipe_names = wn.pipe_name_list
    dfs_list = []

    scenario_metadata = []

    with tqdm(total=len(leak_diameter_parameters)*len(times_of_failure_h)*len(pipe_names), desc="Leak calculations") as pbar:
        for leak_diameter_parameter in leak_diameter_parameters:
            for time_of_failure_h in times_of_failure_h:
                for pipe_to_fail in pipe_names:
                
                    wn = SIMULATION_CONFIG.create_network_real()
                    
                    start_time_s = time_of_failure_h * 3600
                
                    if SIMULATION_CONFIG.time.leak_duration != 0:
                        end_time_s = (time_of_failure_h + SIMULATION_CONFIG.time.leak_duration) * 3600
                    else:
                        end_time_s = SIMULATION_CONFIG.time.duration_s
                
                    pipe = wn.get_link(pipe_to_fail)
                    leak_diameter = pipe.diameter * leak_diameter_parameter
                    leak_area = np.pi * (leak_diameter / 2)**2
                    
                    node_name = pipe_to_fail + '_leak_node'
                    wn = wntr.morph.split_pipe(wn, pipe_to_fail, pipe_to_fail + '_A', node_name)
                    
                    leak_node = wn.get_node(node_name) 

                    wn_base = SIMULATION_CONFIG.create_network_base()
                    sim_base = wntr.sim.WNTRSimulator(wn_base)
                    results_base = sim_base.run_sim()

                    leak_node.add_leak(wn, area=leak_area, start_time=start_time_s, end_time=end_time_s)
                    sim_leak = wntr.sim.WNTRSimulator(wn)
                    results_leak = sim_leak.run_sim()

                    residuals_matrix = results_base.node['pressure'] - results_leak.node['pressure']
                    
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

    metadata_df = pd.DataFrame(scenario_metadata)

    bp_signals = get_blueprint_signals()
    df_combined = pd.merge(bp_signals, signal_final, on=['T', 'Node'], how='outer')

    outlier_scenarios = return_outlier_scenario(df_combined, times_of_failure_h)
    
    if outlier_scenarios is not None:
        metadata_df['is_outlier'] = metadata_df['Scenario_Name'].isin(outlier_scenarios)
        outlier_pipes = metadata_df.loc[metadata_df['is_outlier']]['leak_location'].tolist()
        metadata_df.loc[metadata_df['leak_location'].isin(outlier_pipes), 'is_outlier'] = True

    # metadata_df.to_csv(SIMULATION_CONFIG.output_folder / 'csv' / 'scenario_metadata.csv')
    metadata_df.to_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'scenario_metadata.pkl')

    # df_combined.to_csv(SIMULATION_CONFIG.output_folder / 'csv' / 'signals_with_bp.csv')
    df_combined.to_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'signals_with_bp.pkl')

    return signal_final
 
def run_single_leak(ldp, tofh, pipe_to_fail):

    wn = SIMULATION_CONFIG.create_network_real()
    
    start_time_s = tofh * 3600
    if SIMULATION_CONFIG.time.leak_duration != 0:
        end_time_s = (tofh + SIMULATION_CONFIG.time.leak_duration) * 3600
    else:
        end_time_s = SIMULATION_CONFIG.time.duration_s

    wn.options.hydraulic.demand_model = 'PDD' 

    pipe = wn.get_link(pipe_to_fail)
    leak_diameter = pipe.diameter * ldp
    leak_area = np.pi * (leak_diameter / 2)**2
    
    node_name = f"{pipe_to_fail}_leak_node"
    wn = wntr.morph.split_pipe(wn, pipe_to_fail, pipe_to_fail + '_A', node_name)

    wn_base = SIMULATION_CONFIG.create_network_base()
    sim_base = wntr.sim.WNTRSimulator(wn_base)
    results_base = sim_base.run_sim()

    leak_node = wn.get_node(node_name)
    leak_node.add_leak(wn, 
                       area=leak_area,
                       start_time=start_time_s,
                       end_time=end_time_s)
    
    sim_leak = wntr.sim.WNTRSimulator(wn)
    results_leak = sim_leak.run_sim()

    residuals_matrix = results_base.node['pressure'] - results_leak.node['pressure']

    res_df = residuals_matrix.stack().reset_index()
    res_df.columns = ['T', 'Node', 'Pressure_Residual']
    
    res_df['leak_diameter_parameter'] = ldp
    res_df['time_of_failure_h'] = tofh
    res_df['leak_location'] = pipe_to_fail
    
    return res_df

def stochastic_simulation_signals_parallel(leak_diameter_parameters, times_of_failure_h, n_jobs=-1):

    wn = SIMULATION_CONFIG.create_network_real()
    sim = wntr.sim.WNTRSimulator(wn)
    ref_sim = sim.run_sim()
    ref_pressure_matrix = ref_sim.node['pressure']
    pipe_names = wn.pipe_name_list

    tasks = list(itertools.product(leak_diameter_parameters, times_of_failure_h, pipe_names))

    t_dict = defaultdict(list)

    for index, task in enumerate(tasks):
        t_dict[task].append(index)

    for task, indices in t_dict.items():
        if len(indices) > 1:
            print(f"Tupla {task} występuje na indeksach: {indices}")
    
    results_list = Parallel(n_jobs=n_jobs)(
        delayed(run_single_leak)(ldp, tofh, pipe) 
        for ldp, tofh, pipe in tqdm(tasks, desc="Parallel Leak Calculations")
    )

    results_list = [r for r in results_list if r is not None]
    
    if not results_list:
        print("Błąd: Brak wyników symulacji.")
        return pd.DataFrame(), pd.DataFrame()

    df_long = pd.concat(results_list, axis=0, ignore_index=True)

    df_long['Scenario_Name'] = (
        'ldp' + df_long['leak_diameter_parameter'].astype(str) + 
        '_tofh' + df_long['time_of_failure_h'].astype(str) + 
        '_pl' + df_long['leak_location'] + '_scenario'
    )

    metadata_cols = ['Scenario_Name', 'leak_diameter_parameter', 'time_of_failure_h', 'leak_location']
    scenario_metadata = df_long[metadata_cols].drop_duplicates(subset='Scenario_Name').reset_index(drop=True)

    signal_final = df_long.pivot(
        index=['T', 'Node'], 
        columns='Scenario_Name', 
        values='Pressure_Residual'
    ).reset_index()

    metadata_df = pd.DataFrame(scenario_metadata)

    bp_signals = get_blueprint_signals()
    df_combined = pd.merge(bp_signals, signal_final, on=['T', 'Node'], how='outer')

    # df_combined.to_csv(SIMULATION_CONFIG.output_folder / 'csv' / 'signals_with_bp.csv')
    df_combined.to_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'signals_with_bp.pkl')

    outlier_scenarios = return_outlier_scenario(df_combined, times_of_failure_h)
    
    if outlier_scenarios is not None:
        metadata_df['is_outlier'] = metadata_df['Scenario_Name'].isin(outlier_scenarios)
        outlier_pipes = metadata_df.loc[metadata_df['is_outlier']]['leak_location'].tolist()
        metadata_df.loc[metadata_df['leak_location'].isin(outlier_pipes), 'is_outlier'] = True

    # metadata_df.to_csv(SIMULATION_CONFIG.output_folder / 'csv' / 'scenario_metadata.csv')
    metadata_df.to_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'scenario_metadata.pkl')

    signal_final = signal_final[signal_final['Node'].isin(wn.node_name_list)]

    return signal_final
 