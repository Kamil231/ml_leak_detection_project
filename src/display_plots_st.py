import streamlit as st
import pickle
from src.config import SIMULATION_CONFIG
import wntr
from pathlib import Path
from display_Chama import display_Chama, display_Chama_seperate_leaks
from display_network_map import display_network_map
from display_nodes_parameters import display_nodes_parameters
from display_signals import display_signals
from display_ml_results import display_ml_results

PICKLE_DIR = SIMULATION_CONFIG.output_folder / 'pickle'

@st.cache_data(show_spinner="Ładowanie wszystkich danych wejściowych...")
def load_all_data(base_dir: Path):
    files_to_load = {
        'signals': 'signals_with_bp.pkl',
        'scenario_metadata': 'scenario_metadata.pkl',
        'nodal_thresholds_std': 'nodal_thresholds_std.pkl',
        'sensors_wn_dict': 'sensors_wn_dict.pkl',

        'chama_outputs_single': 'chama_outputs.pkl',
        'chama_outputs_seperate': 'chama_outputs_seperate.pkl',
        'precision_recall_data_chama': 'precision_recall_data_chama.pkl',
        'precision_recall_data_chama_seperate': 'precision_recall_data_chama_seperate.pkl',

        'cm_nn': 'confusion_matrix_df_nn.pkl',
        'cm_xgb': 'confusion_matrix_df_xgb.pkl',
        'cm_lgb': 'confusion_matrix_df_lgb.pkl',

        'cm_best_nn': 'confusion_matrix_best_nodes_df_nn.pkl',
        'cm_best_xgb': 'confusion_matrix_best_nodes_df_xgb.pkl',
        'cm_best_lgb': 'confusion_matrix_best_nodes_df_lgb.pkl',

        'best_nodes_nn': 'best_nodes_nn.pkl',
        'best_nodes_xgb': 'best_nodes_xgb.pkl',
        'best_nodes_gbm': 'best_nodes_lgb.pkl'

    }
    
    loaded_data = {}
    
    for key, filename in files_to_load.items():
        with open(base_dir / filename, 'rb') as file:
            loaded_data[key] = pickle.load(file)
            
    return loaded_data

@st.cache_resource(show_spinner="Uruchamianie i cachowanie symulacji WNTR...")
def load_simulation_results():
    wn_base = SIMULATION_CONFIG.create_network_base()
    wn_real = SIMULATION_CONFIG.create_network_real()
    
    sim_real = wntr.sim.WNTRSimulator(wn_real)
    results_real = sim_real.run_sim()
    
    sim_base = wntr.sim.WNTRSimulator(wn_base)
    results_base = sim_base.run_sim()

    node_name_list = wn_base.node_name_list
    
    return results_real, results_base, node_name_list, wn_real

data = load_all_data(PICKLE_DIR)

chama_outputs_single = data['chama_outputs_single']
chama_outputs_seperate = data['chama_outputs_seperate']
scenario_metadata = data['scenario_metadata']
sensors_wn_dict = data['sensors_wn_dict']
precision_recall_data_chama = data['precision_recall_data_chama']
precision_recall_data_chama_seperate = data['precision_recall_data_chama_seperate']
confusion_matrix_df_XGB = data['cm_xgb']
confusion_matrix_best_nodes_df_XGB = data['cm_best_xgb']
confusion_matrix_df_LGB = data['cm_lgb']
confusion_matrix_best_nodes_df_LGB = data['cm_best_lgb']
df_signals = data['signals']
confusion_matrix_df = data['cm_nn']
confusion_matrix_best_nodes_df = data['cm_best_nn']
nodal_thresholds_std = data['nodal_thresholds_std']
best_nodes_nn = data['best_nodes_nn']
best_nodes_xgb = data['best_nodes_xgb']
best_nodes_gbm = data['best_nodes_gbm']

results_real, results_base, node_name_list, wn = load_simulation_results()

st.set_page_config(layout="wide")

display_signals(scenario_metadata, df_signals, wn, nodal_thresholds_std)

display_nodes_parameters(node_name_list, results_real, results_base)

display_network_map(scenario_metadata, wn)

# display_Chama(chama_outputs, sensors_wn_dict, wn, precision_recall_data_chama)

# display_Chama_seperate_leaks(chama_outputs_seperate, sensors_wn_dict, wn, precision_recall_data_chama_seperate)

display_Chama(chama_outputs_single, sensors_wn_dict, wn, precision_recall_data_chama, chama_outputs_seperate, precision_recall_data_chama_seperate)

display_ml_results(confusion_matrix_df_XGB, confusion_matrix_best_nodes_df_XGB, wn, 'XGB', best_nodes_xgb)

display_ml_results(confusion_matrix_df_LGB, confusion_matrix_best_nodes_df_LGB, wn, 'LGB', best_nodes_gbm)

display_ml_results(confusion_matrix_df, confusion_matrix_best_nodes_df, wn, 'Neural Networks', best_nodes_nn)