from src.config import SIMULATION_CONFIG
import wntr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy


wn_base = SIMULATION_CONFIG.create_network_base()
sim_base = wntr.sim.WNTRSimulator(wn_base)
results_base = sim_base.run_sim()

wn = SIMULATION_CONFIG.create_network_real()
sim_leak = wntr.sim.WNTRSimulator(wn)
results_leak = sim_leak.run_sim()

residuals_matrix = results_base.node['pressure'] - results_leak.node['pressure']

'''res_df = residuals_matrix.stack().reset_index()
res_df.columns = ['T', 'Node', 'Pressure_Residual']'''



val = residuals_matrix.std()

print(val)


