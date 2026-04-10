from src.config import SIMULATION_CONFIG
import wntr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

from src.alter_demand_model import get_alt_demand_wn


def get_wns():

    wn_base = SIMULATION_CONFIG.create_network_base()
    wn_real = SIMULATION_CONFIG.create_network_real()
    
    return wn_base, wn_real

def plot_wns(wn_base, wn_real):
    
    sim_mod = wntr.sim.WNTRSimulator(wn_real)
    results_mod = sim_mod.run_sim()
    demand_mod = results_mod.node['demand']
    
    sim_org = wntr.sim.WNTRSimulator(wn_base)
    results_org = sim_org.run_sim()
    demand_org = results_org.node['demand']
    
    if not demand_mod.index.equals(demand_org.index):
        demand_org = demand_org.reindex(demand_mod.index)
    
    node_name_list = wn_base.node_name_list
    nodes_str = [x for x in node_name_list if x.isdigit()]
    
    for i, node_id in enumerate(nodes_str):
        df_plot = pd.DataFrame(index=demand_mod.index)
        
        df_plot['Modified'] = demand_mod[node_id]
        df_plot['Original'] = demand_org[node_id]
        
        df_plot.set_axis(df_plot.index / 3600).plot(xlabel="hrs", title=f"Node: {node_id}")
        plt.show()
        
  
wn_base, wn_real = get_wns()
plot_wns(copy.deepcopy(wn_base), copy.deepcopy(wn_real))


sim_mod = wntr.sim.WNTRSimulator(wn_real)
results_mod = sim_mod.run_sim()
demand_mod = results_mod.node['demand']

sim_org = wntr.sim.WNTRSimulator(wn_base)
results_org = sim_org.run_sim()
demand_org = results_org.node['demand']
df_plot = pd.DataFrame(index=demand_mod.index)

df_plot['Modified'] = demand_mod['185']
df_plot['Original'] = demand_org['185']
