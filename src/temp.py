from config import SIMULATION_CONFIG
import wntr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

wn = SIMULATION_CONFIG.create_network()
sim = wntr.sim.WNTRSimulator(wn)
results = sim.run_sim()

#%%

p_ref = results.node['pressure']
all_residua = pd.DataFrame()
#%%

p_noisy = copy.deepcopy(results.node['pressure'])

for i in range(SIMULATION_CONFIG.scenarios.sigma3_sim_number):
    try:
        sensor_noise = np.random.normal(0, SIMULATION_CONFIG.scenarios.noise_parameter, size=p_noisy.shape) 
        p_noisy = p_noisy + sensor_noise
    except Exception as e:
        print(f"Błąd symulacji w kroku {i}: {e}")
        continue

all_residua = p_noisy - p_ref

#%%

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
p_ref = p_ref.add_suffix('_ref')
p_noisy = p_noisy.add_suffix('_noisy')
p_plot = pd.concat([p_ref['179_ref'], p_noisy['179_noisy']], axis=1)


print(all_residua['179'].shape)
print(p_ref['179_ref'].shape)
print(p_noisy['179_noisy'].shape)

print(all_residua)
print(p_ref)
print(p_noisy)

ax1.plot(p_plot)
ax1.set_title('signals')

p_plot = pd.concat([p_ref['179_ref'], p_noisy['179_noisy'], all_residua['179']], axis=1)
ax2.plot(p_plot)
ax2.set_title('residuals')

plt.tight_layout()
plt.show()

