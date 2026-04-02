import wntr
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import copy
import pandas as pd
from pprint import pprint
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from src.config import SIMULATION_CONFIG

def generate_random_coeff_list(n, seed):
    
    np.random.seed(seed)

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

def mix_timestamps(demand_node):
    time = demand_node.index.tolist()
    try:
        delta_time = time[1] - time[0]
    except:
        print('cannot calcualte delta time')
        
    demand_node_temp = demand_node
     
    max_shift = 3600 * 6 / delta_time
    
    for i, d in enumerate(demand_node):
        np.random.seed(i)
        random_shift = np.random.randint(max_shift*(-1), max_shift+1)
        '''demand_node.iloc[i] = demand_node_temp.iloc[(i+random_shift)%len(demand_node_temp)]
        print('random_shift: ', demand_node_temp.iloc[(i+random_shift)%len(demand_node_temp)])'''
        
        demand_node.iloc[i] = demand_node_temp.iloc[(i+random_shift)] 
        #print('random_shift: ', demand_node_temp.iloc[(i+random_shift)])
        
        
    return demand_node

def get_alt_demand_wn(wn):
    
    sim = wntr.sim.WNTRSimulator(wn)
    results_real = sim.run_sim()
    demand = results_real.node['demand'].copy()
    
    node_name_list = wn.node_name_list
    nodes_str = [x for x in node_name_list if x.isdigit()]
    nodes_int = [int(x) for x in nodes_str]
    
    for i, node_id in enumerate(nodes_str):
        
        node = wn.get_node(node_id)
        if node.node_type != 'Junction':
            #print(f"ommiting junction {node_id} (type: {node.node_type})")
            continue
        
        coeff_list = generate_random_coeff_list(len(demand), nodes_int[i])        
        demand_shifted = mix_timestamps(demand[node_id].copy())
        #print(f'1. demand_shifted[{node_id}]: ', (demand[node_id] == 0).sum())
        if node_id == '3':
            new_demand_pattern = demand_shifted * coeff_list
            df_to_print = pd.DataFrame({
                'demand_shifted': demand_shifted,
                'coeff_list': coeff_list,
                'new_demand_pattern': new_demand_pattern
                })
        new_demand_pattern = demand_shifted * coeff_list
        new_demand_pattern = new_demand_pattern.tolist()
        demand[node_id] = new_demand_pattern
        node = wn.get_node(node_id)
        try:
            wn.add_pattern(f'RealPattern_{node_id}', new_demand_pattern)
            del node.demand_timeseries_list[:]
            node.add_demand(base=1.0, pattern_name=f'RealPattern_{node_id}')
        except:
            print(f'failed to add: RealPattern_{node_id}')
            continue

    wn.reset_initial_values() 
        
    return wn

def get_wns():

    wn = SIMULATION_CONFIG.create_newtork()
    
    wn_original = copy.deepcopy(wn) 
    
    wn_altered = get_alt_demand_wn(wn) 
    
    return wn_original, wn_altered

def plot_wns(wn_original, wn_altered):
    
    sim_mod = wntr.sim.WNTRSimulator(wn_altered)
    results_mod = sim_mod.run_sim()
    demand_mod = results_mod.node['demand']
    
    sim_org = wntr.sim.WNTRSimulator(wn_original)
    results_org = sim_org.run_sim()
    demand_org = results_org.node['demand']
    
    if not demand_mod.index.equals(demand_org.index):
        demand_org = demand_org.reindex(demand_mod.index)
    
    node_name_list = wn_original.node_name_list
    nodes_str = [x for x in node_name_list if x.isdigit()]
    
    for i, node_id in enumerate(nodes_str):
        df_plot = pd.DataFrame(index=demand_mod.index)
        
        df_plot['Modified'] = demand_mod[node_id]
        df_plot['Original'] = demand_org[node_id]
        
        df_plot.set_axis(df_plot.index / 3600).plot(xlabel="hrs", title=f"Node: {node_id}")
        plt.show()
        
  
'''wn_original, wn_altered = get_wns()
plot_wns(copy.deepcopy(wn_original), copy.deepcopy(wn_altered))


sim_mod = wntr.sim.WNTRSimulator(wn_altered)
results_mod = sim_mod.run_sim()
demand_mod = results_mod.node['demand']

sim_org = wntr.sim.WNTRSimulator(wn_original)
results_org = sim_org.run_sim()
demand_org = results_org.node['demand']
df_plot = pd.DataFrame(index=demand_mod.index)

df_plot['Modified'] = demand_mod['185']
df_plot['Original'] = demand_org['185']

print(df_plot)'''

