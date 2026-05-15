import wntr
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import copy
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

def generate_random_coeff_list(n, seed, seed_offset = 0):
    
    np.random.seed(seed + seed_offset)

    random_coeff_ranges = np.random.choice(
        [0, 1, 2], 
        size=n, 
        p=[35/169, 22/169, 112/169]
    )
    
    results = np.zeros(n)
    
    mask1 = (random_coeff_ranges == 1)
    results[mask1] = np.random.uniform(1.0, 1.8, size=mask1.sum())
    
    mask2 = (random_coeff_ranges == 2)
    results[mask2] = np.random.uniform(0.2, 0.8, size=mask2.sum())
    
    return results

def mix_timestamps(demand_node, seed_offset = 0):
    time = demand_node.index.tolist()
    try:
        delta_time = time[1] - time[0]
    except Exception as e:
        print(f'Cannot calculate delta time: {e}')
        return demand_node
        
    demand_node_temp = copy.deepcopy(demand_node)
     
    max_shift = int(3600 * 6 / delta_time)

    n_elements = len(demand_node)
    
    for i in range(n_elements):
        np.random.seed(i + seed_offset * 10000) # x 10000 zeby zierna sie nie nakladaly
        random_shift = np.random.randint(max_shift*(-1), max_shift+1)
        target_idx = max(0, min(i + random_shift, n_elements - 1))        
        demand_node.iloc[i] = demand_node_temp.iloc[target_idx]
        
    return demand_node

def get_alt_demand_wn(wn, seed_offset = 0):
    
    sim = wntr.sim.WNTRSimulator(wn)
    results_real = sim.run_sim()
    demand = results_real.node['demand'].copy()
    
    node_name_list = wn.node_name_list
    nodes_str = [x for x in node_name_list if x.isdigit()]
    nodes_int = [int(x) for x in nodes_str]
    
    for i, node_id in enumerate(nodes_str):
        
        node = wn.get_node(node_id)
        
        if node.node_type != 'Junction':
            continue

        coeff_list = generate_random_coeff_list(len(demand), nodes_int[i], seed_offset)        
        demand_shifted = mix_timestamps(demand[node_id].copy(), seed_offset)

        new_demand_pattern = demand_shifted * coeff_list
        new_demand_pattern = new_demand_pattern.tolist()
        demand[node_id] = new_demand_pattern
        node = wn.get_node(node_id)

        try:
            wn.add_pattern(f'RealPattern_{node_id}', new_demand_pattern)
            del node.demand_timeseries_list[:]
            node.add_demand(base=1.0, pattern_name=f'RealPattern_{node_id}')
        except Exception as e:
            print(f'{i}/{len(nodes_str)} failed to add: RealPattern_{node_id}')
            print(f"Wystąpił błąd: {e}")
            continue

    wn.reset_initial_values() 
        
    return wn





