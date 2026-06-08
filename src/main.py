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
from src.run_chama_analysis import run_chama_simulation, run_chama_simulation_parallel
import logging
import numpy as np
import yaml
import time
from dataclasses import asdict
from pathlib import Path
from src.pytorch_nn import nn_analysis_parallel, NN_analysis_best_nodes
from src.LightGBM_analysis import LightGBM_analysis_all_nodes, LightGBM_analysis_best_nodes
from src.XGBoost_analysis import XGBoost_analysis_all_nodes, XGBoost_analysis_best_nodes
from src.generate_ml_dataset import get_signals_df


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


if __name__ == '__main__':

    start_time = time.perf_counter()

    # leak_diameter_parameters = [0.01, .05, .1, 0.15, .02, 0.4, 0.6]
    # leak_diameter_parameters = [0.1, 0.6]
    leak_diameter_parameters = [round(x * 0.05, 2) for x in range(1, 20)]
    times_of_failure_h = [24, 48]
    # times_of_failure_h = [48]
    print('combination number: ', len(leak_diameter_parameters)*len(times_of_failure_h))

    # run_chama_simulation(leak_diameter_parameters, times_of_failure_h)
    run_chama_simulation_parallel(leak_diameter_parameters, times_of_failure_h)

    get_signals_df()

    XGBoost_analysis_all_nodes()

    XGBoost_analysis_best_nodes()

    LightGBM_analysis_all_nodes()

    LightGBM_analysis_best_nodes()

    nn_analysis_parallel()
    
    NN_analysis_best_nodes()

    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print(f"Czas wykonania: {elapsed_time:.4f} sekund")

    with open(SIMULATION_CONFIG.output_folder / 'config.txt', 'w', encoding='utf-8') as f:
        config_dict = asdict(SIMULATION_CONFIG)
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


