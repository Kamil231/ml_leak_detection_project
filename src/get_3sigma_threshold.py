import numpy as np
import wntr
import pandas as pd
from src.config import SIMULATION_CONFIG
from tqdm import tqdm

def get_1sigma_threshold(num_runs=100):

    wn_base = SIMULATION_CONFIG.create_network_base()
    sim_base = wntr.sim.WNTRSimulator(wn_base)
    results_base = sim_base.run_sim()
    base_pressure = results_base.node['pressure']

    all_residuals = []

    for i in tqdm(range(num_runs), desc="Calculating thresholds: "):

        wn_real = SIMULATION_CONFIG.create_network_real(seed_offset=i)
        sim_real = wntr.sim.WNTRSimulator(wn_real)
        results_real = sim_real.run_sim()

        real_pressure = results_real.node['pressure']

        residuals_matrix = base_pressure - real_pressure
        all_residuals.append(residuals_matrix)

    combined_residuals = pd.concat(all_residuals, ignore_index=True)

    thresholds_std = combined_residuals.std()
    thresholds_mean = combined_residuals.mean()

    df_thresholds = pd.DataFrame({
        'mean': thresholds_mean,
        'std': thresholds_std
    })

    df_thresholds.to_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'nodal_thresholds_std.pkl')

    return df_thresholds


