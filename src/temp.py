from src.config import SIMULATION_CONFIG
import wntr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



wn = SIMULATION_CONFIG.create_newtork()
sim = wntr.sim.WNTRSimulator(wn)
results = sim.run_sim()

#%%

node_names = wn.node_name_list
print("Wszystkie węzły:", node_names)

#%%

p_ref = results.node['pressure']
all_residua = []

for i in range(SIMULATION_CONFIG.scenarios.sigma3_sim_number):
    wn_temp = SIMULATION_CONFIG.create_newtork() 
    for name, node in wn_temp.nodes.junctions():
        #node.demand_timeseries_list[0].base_value *= np.random.uniform(0.95, 1.05)
        node.demand_timeseries_list[0].base_value *= np.random.uniform(1 - SIMULATION_CONFIG.scenarios.demand_noise_parameter, 1 + SIMULATION_CONFIG.scenarios.demand_noise_parameter)
        
    try:
        res = wntr.sim.WNTRSimulator(wn_temp).run_sim()
        p_noisy = res.node['pressure']
        sensor_noise = np.random.normal(0, 4, size=p_noisy.shape) 
        p_noisy = p_noisy + sensor_noise
    except Exception as e:
        print(f"Błąd symulacji w kroku {i}: {e}")
        continue




 
#%%

print(p_noisy)
print(p_ref) 

p_ref = p_ref.add_suffix('_ref')
p_noisy = p_noisy.add_suffix('_noisy')

# p_plot = pd.concat([p_ref['20_ref'], p_noisy['20_noisy']], axis=1)
# p_plot = pd.concat([p_ref['50_ref'], p_noisy['50_noisy']], axis=1)
p_plot = pd.concat([p_ref['179_ref'], p_noisy['179_noisy']], axis=1)

ax = p_plot.plot() 
ax.legend(loc='best')
plt.show()

plt.close("all")