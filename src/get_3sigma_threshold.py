import numpy as np
import wntr
import pandas as pd
from src.config import SIMULATION_CONFIG
from tqdm import tqdm

def get_1sigma_threshold():

    # np.random.seed(42)

    # wn = SIMULATION_CONFIG.create_newtork()
    # sim_ideal = wntr.sim.WNTRSimulator(wn)
    # res_ideal = sim_ideal.run_sim()
    # p_ideal = res_ideal.node['pressure']

    # all_residua = []

    # for i in range(SIMULATION_CONFIG.scenarios.sigma3_sim_number):
    #     wn_temp = SIMULATION_CONFIG.create_newtork() 
    #     for name, node in wn_temp.nodes.junctions():
    #         #node.demand_timeseries_list[0].base_value *= np.random.uniform(0.95, 1.05)
    #         node.demand_timeseries_list[0].base_value *= np.random.uniform(1 - SIMULATION_CONFIG.scenarios.demand_noise_parameter, 1 + SIMULATION_CONFIG.scenarios.demand_noise_parameter)
            
    #     try:
    #         res = wntr.sim.WNTRSimulator(wn_temp).run_sim()
    #         p_noisy = res.node['pressure']
    #         sensor_noise = np.random.normal(0, SIMULATION_CONFIG.scenarios.sensor_noise_parameter, size=p_noisy.shape) 
    #         p_noisy_with_sensor = p_noisy + sensor_noise
    #         all_residua.append((p_noisy_with_sensor - p_ideal))
    #     except Exception as e:
    #         print(f"Błąd symulacji w kroku {i}: {e}")
    #         continue

    # full_res_df = pd.concat(all_residua)
    # thresholds = full_res_df.std()

    wn_base = SIMULATION_CONFIG.create_network_base()
    sim_base = wntr.sim.WNTRSimulator(wn_base)
    results_base = sim_base.run_sim()

    wn = SIMULATION_CONFIG.create_network_real()
    sim_leak = wntr.sim.WNTRSimulator(wn)
    results_leak = sim_leak.run_sim()

    residuals_matrix = results_base.node['pressure'] - results_leak.node['pressure']

    thresholds = residuals_matrix.std()

    return thresholds

def get_xsigma_threshold_dict(threshold_parameters):

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
    
    detectability_matrix.to_csv(SIMULATION_CONFIG.output_folder + '/detectability_matrix.csv')

