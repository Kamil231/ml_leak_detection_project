import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='wntr')
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="chama")
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
import matplotlib.pyplot as plt
from src.config import SIMULATION_CONFIG
import os
import itertools
from src.stochastic_simulation_signals import stochastic_simulation_signals, stochastic_simulation_signals_parallel
import src.sensors_ImpactFormulation as sensors_ImpactFormulation
import src.sensors_CoverageFormulation as sensors_CoverageFormulation
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
import streamlit as st
import pandas as pd
import plotly.express as px
import pprint
import pickle
from joblib import Parallel, delayed
from pathlib import Path
from tqdm_joblib import tqdm_joblib
import wntr
from src.get_3sigma_threshold import get_1sigma_threshold
from src.precision_recall import get_precision_recall_data, get_precision_recall_data_seperate

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

def run_chama_simulation(leak_diameter_parameters, times_of_failure_h):

    threshold_parameters = [3]

    os.makedirs(SIMULATION_CONFIG.output_folder / 'pickle', exist_ok=True)   
    os.makedirs(SIMULATION_CONFIG.output_folder / 'csv', exist_ok=True)    

    chama_outputs = {}
    sensors_thp_dict = {}

    signal = stochastic_simulation_signals(leak_diameter_parameters, times_of_failure_h)  
    signal_input = signal.copy()

    scenario_metadata = pd.read_pickle(SIMULATION_CONFIG.output_folder / 'pickle/scenario_metadata.pkl')   
    cols_to_drop = scenario_metadata.loc[scenario_metadata['is_outlier']]['Scenario_Name'].tolist()
    signal_input = signal_input.drop(columns=cols_to_drop)

    thresholds_series = get_1sigma_threshold()

    for sensor_budget in tqdm(SIMULATION_CONFIG.scenarios.sensor_budget):

        wn_local = SIMULATION_CONFIG.create_network_real()
        results_ImpactFormulation, sensors_thp_dict = sensors_ImpactFormulation.get_sensor_locations(wn_local, signal_input, threshold_parameters, thresholds_series, sensor_budget)
        wn_local = SIMULATION_CONFIG.create_network_real()
        results_CoverageFormulation, sensors_thp_dict = sensors_CoverageFormulation.get_sensor_locations(wn_local, signal_input, threshold_parameters, thresholds_series, sensor_budget)
        
        chama_outputs[sensor_budget] = (results_ImpactFormulation, results_CoverageFormulation)

    chama_outputs_temp_list = [] 
    for budget, formulations in chama_outputs.items():

        chama_outputs_temp_list.append({
            'Budget': budget,
            'Formulation': 'ImpactFormulation',
            'Result': formulations[0]
        })

        chama_outputs_temp_list.append({
            'Budget': budget,
            'Formulation': 'CoverageFormulation',
            'Result': formulations[1]
        })

    chama_outputs = pd.DataFrame(chama_outputs_temp_list)


    with open(SIMULATION_CONFIG.output_folder / 'pickle' / 'chama_outputs.pkl', 'wb') as f:
        pickle.dump(chama_outputs, f)

    with open(SIMULATION_CONFIG.output_folder / 'pickle' / 'sensors_thp_dict.pkl', 'wb') as f:
        pickle.dump(sensors_thp_dict, f)

    precision_recall_data = get_precision_recall_data()
    precision_recall_data.to_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'precision_recall_data_chama.pkl')

def run_chama_simulation_parallel(leak_diameter_parameters, times_of_failure_h):
    threshold_parameters = [3]
    output_base = Path(SIMULATION_CONFIG.output_folder)
    pickle_path = output_base / "pickle"
    csv_path = output_base / "csv"
    # pickle_path.mkdir(parents=True, exist_ok=True)
    # csv_path.mkdir(parents=True, exist_ok=True)

    thresholds_series = get_1sigma_threshold()

    # nodal_thresholds_dict = get_xsigma_threshold_dict(threshold_parameters)
    # pd.DataFrame(nodal_thresholds_dict).to_csv(csv_path / "nodal_thresholds.csv")

    signal = stochastic_simulation_signals_parallel(leak_diameter_parameters, times_of_failure_h)
    #signal_input = signal.drop(columns=['leak_diameter_parameter', 'time_of_failure_h']).copy()
    signal_input = signal.copy()

    scenario_metadata = pd.read_pickle(SIMULATION_CONFIG.output_folder / 'pickle/scenario_metadata.pkl')   
    cols_to_drop = scenario_metadata.loc[scenario_metadata['is_outlier']]['Scenario_Name'].tolist()
    signal_input = signal_input.drop(columns=cols_to_drop)

    if signal_input.empty:
        print("WARNING: signal_input is empty! Optimization will do nothing.")
        return

    def optimize_for_budget(n):
        
        warnings.filterwarnings("ignore")
        wn_local = SIMULATION_CONFIG.create_network_real()
        res_impact, s_dict_i = sensors_ImpactFormulation.get_sensor_locations(
            wn_local, signal_input, threshold_parameters, thresholds_series, n
        )
        res_coverage, s_dict_c = sensors_CoverageFormulation.get_sensor_locations(
            wn_local, signal_input, threshold_parameters, thresholds_series, n
        )
        return n, res_impact, res_coverage, {**s_dict_i, **s_dict_c}

    budgets = SIMULATION_CONFIG.scenarios.sensor_budget
    parallel_pool = Parallel(n_jobs=4)

    parallel_results = []

    with tqdm_joblib(tqdm(desc="Real-time Optimization Progress", total=len(budgets))) as pbar:
        job_generator = (delayed(optimize_for_budget)(n) for n in budgets)
        parallel_results = parallel_pool(job_generator)


    chama_outputs = {}
    final_sensors_thp_dict = {}

    for n, res_i, res_c, s_dict in parallel_results:
        chama_outputs[n] = (res_i, res_c)
        final_sensors_thp_dict.update(s_dict)


    chama_outputs_temp_list = [] 
    for budget, formulations in chama_outputs.items():

        chama_outputs_temp_list.append({
            'Budget': budget,
            'Formulation': 'ImpactFormulation',
            'Result': formulations[0]
        })

        chama_outputs_temp_list.append({
            'Budget': budget,
            'Formulation': 'CoverageFormulation',
            'Result': formulations[1]
        })

    chama_outputs = pd.DataFrame(chama_outputs_temp_list)

    with open(pickle_path / "chama_outputs.pkl", 'wb') as f:
        pickle.dump(chama_outputs, f)
    with open(pickle_path / "sensors_thp_dict.pkl", 'wb') as f:
        pickle.dump(final_sensors_thp_dict, f)

    precision_recall_data = get_precision_recall_data()
    precision_recall_data.to_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'precision_recall_data_chama.pkl')

def run_chama_simulation_parallel_seperate_leaks(leak_diameter_parameters, times_of_failure_h):
    threshold_parameters = [3]
    output_base = Path(SIMULATION_CONFIG.output_folder)
    pickle_path = output_base / "pickle"

    thresholds_series = get_1sigma_threshold()

    #signal = stochastic_simulation_signals_parallel(leak_diameter_parameters, times_of_failure_h)
    signal = pd.read_pickle(pickle_path / 'signals_with_bp.pkl')
    signal = signal.drop(columns=['blueprint_scenario'])
    signal_input_full = signal.copy()

    scenario_metadata = pd.read_pickle(pickle_path / 'scenario_metadata.pkl')   

    cols_to_drop = scenario_metadata.loc[scenario_metadata['is_outlier']]['Scenario_Name'].tolist()
    signal_input_full = signal_input_full.drop(columns=[c for c in cols_to_drop if c in signal_input_full.columns])

    if signal_input_full.empty:
        print("WARNING: signal_input is empty! Optimization will do nothing.")
        return

    def optimize_for_budget_ldp(n, sig_in):
        warnings.filterwarnings("ignore")
        wn_local = SIMULATION_CONFIG.create_network_real()
        res_impact, s_dict_i = sensors_ImpactFormulation.get_sensor_locations(
            wn_local, sig_in, threshold_parameters, thresholds_series, n
        )
        res_coverage, s_dict_c = sensors_CoverageFormulation.get_sensor_locations(
            wn_local, sig_in, threshold_parameters, thresholds_series, n
        )
        return n, res_impact, res_coverage, {**s_dict_i, **s_dict_c}

    budgets = SIMULATION_CONFIG.scenarios.sensor_budget
    chama_outputs_temp_list = []
    final_sensors_thp_dict = {}

    for ldp in leak_diameter_parameters:
        print(f"\nRozpoczynam optymalizację Chama dla średnicy wycieku: {ldp}")

        valid_scenarios = scenario_metadata[
            (scenario_metadata['leak_diameter_parameter'] == ldp) & 
            (scenario_metadata['is_outlier'] == False)
        ]['Scenario_Name'].tolist()
        
        cols_to_keep = ['T', 'Node'] + [s for s in valid_scenarios if s in signal_input_full.columns]
        signal_input_ldp = signal_input_full[cols_to_keep].copy()

        if len(cols_to_keep) <= 2:
            print(f"WARNING: Brak prawidłowych scenariuszy dla ldp={ldp}. Pomijam.")
            continue

        parallel_pool = Parallel(n_jobs=4)

        with tqdm_joblib(tqdm(desc=f"Optymalizacja budżetów (LDP={ldp})", total=len(budgets))) as pbar:
            job_generator = (delayed(optimize_for_budget_ldp)(n, signal_input_ldp) for n in budgets)
            parallel_results = parallel_pool(job_generator)

        for n, res_i, res_c, s_dict in parallel_results:
            chama_outputs_temp_list.append({
                'Leak_Diameter': ldp, 
                'Budget': n,
                'Formulation': 'ImpactFormulation',
                'Result': res_i
            })
            chama_outputs_temp_list.append({
                'Leak_Diameter': ldp,
                'Budget': n,
                'Formulation': 'CoverageFormulation',
                'Result': res_c
            })
            final_sensors_thp_dict.update(s_dict)

    chama_outputs_seperate = pd.DataFrame(chama_outputs_temp_list)

    with open(pickle_path / "chama_outputs_seperate.pkl", 'wb') as f:
        pickle.dump(chama_outputs_seperate, f)
    with open(pickle_path / "sensors_thp_dict_seperate.pkl", 'wb') as f:
        pickle.dump(final_sensors_thp_dict, f)

    pr_data_seperate = get_precision_recall_data_seperate()
    pr_data_seperate.to_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'precision_recall_data_chama_seperate.pkl')
        