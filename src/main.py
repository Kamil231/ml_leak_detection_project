import os
os.environ["PYTHONWARNINGS"] = "ignore"
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

from src.config import SIMULATION_CONFIG
from src.run_chama_analysis import run_simulation, run_simulation_parallel
import logging
import numpy as np
import yaml
from dataclasses import asdict
from pathlib import Path

#LOG_FILE = 'sim_warnings.log'

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
log_dir = project_root / 'logs'
LOG_FILE = log_dir / 'sim_warnings.log'


logger = logging.getLogger('wntr_chama_sim')
logger.setLevel(logging.WARNING)
file_handler = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
file_handler.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logging.captureWarnings(True)



#threshold_parameters = np.arange(1, 3.1, 0.1).tolist()
#threshold_parameters = np.round(np.arange(1, 3.1, 0.1), 2).tolist()
#print('threshold_parameters: ', threshold_parameters)
'''threshold_parameters = [3, 2.8, 2.6, 2.4, 2.2, 2, 1.8, 1.6, 1.4, 1.2, 1]'''
'''leak_diameter_parameters = [0.8, 0.6, 0.4, 0.3, 0.25, 0.2, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.1, 0.09, 0.08, 0.07, 0.05]
times_of_failure_h = [8, 16, 24]
'''


#leak_diameter_parameter = 0.6, time_of_failure_h = 24, threshold_parameter = 3
#leak diameter parameter = 0.3, time of failure h = 16, threshold parameter = 2.6

'''threshold_parameters = [2.8]
leak_diameter_parameters = [0.6]
times_of_failure_h = [8]'''






# threshold_parameters = [1]
# leak_diameter_parameters = [0.05]
# times_of_failure_h = [0]
# print('threshold_parameters: ', len(threshold_parameters))
# print('combination number: ', len(leak_diameter_parameters)*len(times_of_failure_h))
# #run_simulation(leak_diameter_parameters, times_of_failure_h, threshold_parameters)
# run_simulation_parallel(leak_diameter_parameters, times_of_failure_h, threshold_parameters)

threshold_parameters = [1, 2]
leak_diameter_parameters = [0.1, 0.6]
times_of_failure_h = [0, 8]
print('threshold_parameters: ', len(threshold_parameters))
print('combination number: ', len(leak_diameter_parameters)*len(times_of_failure_h))
#run_simulation(leak_diameter_parameters, times_of_failure_h, threshold_parameters)
run_simulation_parallel(leak_diameter_parameters, times_of_failure_h, threshold_parameters)

# threshold_parameters = [1, 2]
# leak_diameter_parameters = [0.05, 0.1, 0.6]
# times_of_failure_h = [0, 8]
# print('threshold_parameters: ', len(threshold_parameters))
# print('combination number: ', len(leak_diameter_parameters)*len(times_of_failure_h))
# #run_simulation(leak_diameter_parameters, times_of_failure_h, threshold_parameters)
# run_simulation_parallel(leak_diameter_parameters, times_of_failure_h, threshold_parameters)

# threshold_parameters = [1, 2, 3]
# leak_diameter_parameters = [0, 0.05, 0.1, 0.6]
# times_of_failure_h = [16, 30]
# print('threshold_parameters: ', len(threshold_parameters))
# print('combination number: ', len(leak_diameter_parameters)*len(times_of_failure_h))
# #run_simulation(leak_diameter_parameters, times_of_failure_h, threshold_parameters)
# run_simulation_parallel(leak_diameter_parameters, times_of_failure_h, threshold_parameters)

# threshold_parameters = [1, 1.5, 2, 2.5, 3]
# leak_diameter_parameters = [0.05, 0.75, 0.1, 0.125, 0.15, 0.2, 0.4, 0.6]
# times_of_failure_h = [0, 8, 16]
# print('threshold_parameters: ', len(threshold_parameters))
# print('combination number: ', len(leak_diameter_parameters)*len(times_of_failure_h))
# print(threshold_parameters)
# print(leak_diameter_parameters)
# print(times_of_failure_h)
# #run_simulation(leak_diameter_parameters, times_of_failure_h, threshold_parameters)
# run_simulation_parallel(leak_diameter_parameters, times_of_failure_h, threshold_parameters)

# #Big Dataset
#threshold_parameters = [1, 1.5, 2, 2.5, 3]
# threshold_parameters = [3]
# leak_diameter_parameters = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.4, 0.6]
# times_of_failure_h = [0, 8]
# print('threshold_parameters: ', len(threshold_parameters))
# print('combination number: ', len(leak_diameter_parameters)*len(times_of_failure_h))
# print(threshold_parameters)
# print(leak_diameter_parameters)
# print(times_of_failure_h)
#run_simulation(leak_diameter_parameters, times_of_failure_h, threshold_parameters)
#run_simulation_parallel(leak_diameter_parameters, times_of_failure_h, threshold_parameters)


with open(SIMULATION_CONFIG.output_folder / 'config.txt', 'w', encoding='utf-8') as f:
    config_dict = asdict(SIMULATION_CONFIG)
    yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


# (praca_mgr_v4) kamilzawitaj@Mac Ex15 % python main.py
# threshold_parameters:  1
# combination number:  1
# --- Generating signals ---
# tasks[1]:  (0.05, 0, '40')
# type(tasks[1]):  <class 'tuple'>
# Parallel Leak Calculations: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 117/117 [00:08<00:00, 13.82it/s]
# Signal shape: (9409, 119)
# Real-time Optimization Progress: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:52<00:00,  5.22s/it]
# --- Completed. Results in: /Users/kamilzawitaj/Documents/Studia/PW OKNO - AiR/Praca Mgr/Code/Ex15/output_folder ---
# threshold_parameters:  2
# combination number:  4
# --- Generating signals ---
# tasks[1]:  (0.1, 0, '40')
# type(tasks[1]):  <class 'tuple'>
# Parallel Leak Calculations: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 468/468 [00:30<00:00, 15.13it/s]
# Signal shape: (9409, 470)
# Real-time Optimization Progress: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [04:35<00:00, 27.60s/it]
# --- Completed. Results in: /Users/kamilzawitaj/Documents/Studia/PW OKNO - AiR/Praca Mgr/Code/Ex15/output_folder ---
# threshold_parameters:  2
# combination number:  6
# --- Generating signals ---
# tasks[1]:  (0.05, 0, '40')
# type(tasks[1]):  <class 'tuple'>
# Parallel Leak Calculations: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 702/702 [00:53<00:00, 13.10it/s]
# Signal shape: (9409, 704)
# Real-time Optimization Progress:   0%|                                                                                                                                                                                                                  | 0/10 [00:00<?, ?it/s]/opt/miniconda3/envs/praca_mgr_v4/lib/python3.11/site-packages/wntr/epanet/toolkit.py:14: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
#   from pkg_resources import resource_filename
# Real-time Optimization Progress: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [06:58<00:00, 41.90s/it]
# --- Completed. Results in: /Users/kamilzawitaj/Documents/Studia/PW OKNO - AiR/Praca Mgr/Code/Ex15/output_folder ---
# threshold_parameters:  3
# combination number:  6
# --- Generating signals ---
# tasks[1]:  (0.05, 0, '40')
# type(tasks[1]):  <class 'tuple'>
# Parallel Leak Calculations: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 702/702 [00:51<00:00, 13.57it/s]
# Signal shape: (9409, 704)
# Real-time Optimization Progress:   0%|                                                                                                                                                                                                                  | 0/10 [00:00<?, ?it/s]/opt/miniconda3/envs/praca_mgr_v4/lib/python3.11/site-packages/wntr/epanet/toolkit.py:14: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
#   from pkg_resources import resource_filename
# Real-time Optimization Progress: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [09:48<00:00, 58.85s/it]
# --- Completed. Results in: /Users/kamilzawitaj/Documents/Studia/PW OKNO - AiR/Praca Mgr/Code/Ex15/output_folder ---

