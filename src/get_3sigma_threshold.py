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

    #usrednione przebiegi z rzeczywistykm demnad
    for i in tqdm(range(num_runs), desc="Calculating thresholds: "):

        wn_real = SIMULATION_CONFIG.create_network_real(seed_offset=i)
        sim_real = wntr.sim.WNTRSimulator(wn_real)
        results_real = sim_real.run_sim()

        real_pressure = results_real.node['pressure']

        residuals_matrix = base_pressure - real_pressure
        all_residuals.append(residuals_matrix)

    combined_residuals = pd.concat(all_residuals, ignore_index=True)
    thresholds = combined_residuals.std()

    thresholds.to_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'nodal_thresholds_std.pkl')

    return thresholds


'''def get_xsigma_threshold_dict(threshold_parameters):

    sigma = get_1sigma_threshold()

    nodal_thresholds_dict = {}
    for threshold_parameter in threshold_parameters:
        nodal_thresholds_dict[threshold_parameter] = sigma * threshold_parameter

    return nodal_thresholds_dict
    
def get_detectability_matrix(wn, threshold_parameter):
    
    base_simulation = []
    
    for i in range(SIMULATION_CONFIG.scenarios.sigma3_sim_number):
        for name, node in wn.nodes.junctions():
            node.demand_timeseries_list[0].base_value *= np.random.uniform(0.9, 1.1)
        
        sim = wntr.sim.WNTRSimulator(wn)
        res = sim.run_sim()
        base_simulation.append(res.node['pressure'])
    
    full_base_df = pd.concat(base_simulation)
    sigmas = full_base_df.std() 
    thresholds = sigmas * threshold_parameter
    detectability_matrix = pd.DataFrame(index=wn.junction_name_list, columns=wn.junction_name_list)

    for leak_node in wn.junction_name_list:
        wn_leak = SIMULATION_CONFIG.create_newtork()
        node = wn_leak.get_node(leak_node)
        node.add_leak(wn_leak, area=0.02, start_time=3600)
        
        sim = wntr.sim.WNTRSimulator(wn_leak)
        res_leak = sim.run_sim()
        pressure_leak = res_leak.node['pressure'].mean() 
        pressure_base = full_base_df.mean()           

        pressure_drop = pressure_base - pressure_leak
        detectability_matrix[leak_node] = (pressure_drop > thresholds).astype(int)
    
    detectability_matrix.to_csv(SIMULATION_CONFIG.output_folder + '/detectability_matrix.csv')'''

