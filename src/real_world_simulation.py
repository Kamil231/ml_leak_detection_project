import wntr

def real_world_simulation(wn_ref):

	sim_ref = wntr.sim.WNTRSimulator(wn_ref)
    results_ref = sim_ref.run_sim()

    return results_ref


def get_1sigma_threshold():

    np.random.seed(42)

    wn = SIMULATION_CONFIG.create_network()
    sim_ideal = wntr.sim.WNTRSimulator(wn)
    res_ideal = sim_ideal.run_sim()
    p_ideal = res_ideal.node['pressure']

    all_residua = []

    for i in range(SIMULATION_CONFIG.scenarios.sigma3_sim_number):
        wn_temp = SIMULATION_CONFIG.create_network() 
        for name, node in wn_temp.nodes.junctions():
            #node.demand_timeseries_list[0].base_value *= np.random.uniform(0.95, 1.05)
            node.demand_timeseries_list[0].base_value *= np.random.uniform(1 - SIMULATION_CONFIG.scenarios.demand_noise_parameter, 1 + SIMULATION_CONFIG.scenarios.demand_noise_parameter)
            
        try:
            res = wntr.sim.WNTRSimulator(wn_temp).run_sim()
            p_noisy = res.node['pressure']
            sensor_noise = np.random.normal(0, SIMULATION_CONFIG.scenarios.sensor_noise_parameter, size=p_noisy.shape) 
            p_noisy_with_sensor = p_noisy + sensor_noise
            all_residua.append((p_noisy_with_sensor - p_ideal))
        except Exception as e:
            print(f"Błąd symulacji w kroku {i}: {e}")
            continue

    full_res_df = pd.concat(all_residua)
    thresholds = full_res_df.std()

    return thresholds
