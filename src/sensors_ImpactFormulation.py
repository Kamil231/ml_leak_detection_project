import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='wntr')
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="chama")
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
import numpy as np
import wntr
import pandas as pd
import matplotlib.pylab as plt
import warnings
from src.config import SIMULATION_CONFIG
import chama
import os
from src.get_3sigma_threshold import get_1sigma_threshold
import pickle

def get_sensor_locations(wn, signal, threshold_parameters, thresholds_series, sensor_budget):

    scenario_names = [col for col in signal.columns if col not in ['T', 'Node']]
    sensor_names = wn.junction_name_list
    sample_times = np.arange(0, wn.options.time.duration, wn.options.time.hydraulic_timestep)

    sensors = {}
    sensors_thp_dict = {}
    grouped_sensors_list = []
    cost_data_list = []

    sensors_wn_dict = {}

    for location in sensor_names:
        position = chama.sensors.Stationary(location)
        same_location_sensors = []
        m_node = thresholds_series.loc[location, 'mean']
        s_node = thresholds_series.loc[location, 'std']
        for threshold_parameter in threshold_parameters:
            sensor_name = f'Node{location}_thp{threshold_parameter}'
            sensors_thp_dict[sensor_name] = threshold_parameter

            node_threshold = m_node + (threshold_parameter * s_node)
            
            detector = chama.sensors.Point(node_threshold, sample_times)
            stationary_pt_sensor = chama.sensors.Sensor(position, detector)
            sensors[sensor_name] = stationary_pt_sensor
            same_location_sensors.append(sensor_name)
            cost_data_list.append({'Sensor': sensor_name, 'Cost': 1})
            sensors_wn_dict[sensor_name] = (location, threshold_parameter)
        grouped_sensors_list.append(same_location_sensors)

    with open(SIMULATION_CONFIG.output_folder  / 'pickle' / 'sensors_wn_dict.pkl', 'wb') as f:
        pickle.dump(sensors_wn_dict, f)

    with open(SIMULATION_CONFIG.output_folder  / 'pickle' / 'sensors_thp_dict.pkl', 'wb') as f:
        pickle.dump(sensors_thp_dict, f)

    det_times = chama.impact.extract_detection_times(signal, sensors)

    det_time_stats = chama.impact.detection_time_stats(det_times)

    min_det_time = det_time_stats.reset_index()
    min_det_time = min_det_time[['Scenario', 'Sensor', 'Min']].copy()
    min_det_time['Scenario'] = min_det_time['Scenario'].astype(object)
    min_det_time['Sensor'] = min_det_time['Sensor'].astype(object)
    min_det_time.rename(columns = {'Min':'Impact'}, inplace = True)

    #min_det_time.to_csv(SIMULATION_CONFIG.output_folder + '/csv/min_det_time.csv')

    scenario_characteristics = pd.DataFrame({'Scenario': scenario_names,
                                             'Undetected Impact': sample_times.max()*1.5})

    sensor_characteristics = pd.DataFrame(cost_data_list)

    impactform = chama.optimize.ImpactFormulation()
    model = impactform.create_pyomo_model(impact=min_det_time, sensor=sensor_characteristics, scenario=scenario_characteristics)

    valid_sensors = set(min_det_time['Sensor'])

    for sensor_group in grouped_sensors_list:
        filtered_group = [s for s in sensor_group if s in valid_sensors]
        if filtered_group:
            impactform.add_grouping_constraint(filtered_group, max_select=1)
        
    impactform.solve_pyomo_model(sensor_budget=sensor_budget)

    results = impactform.create_solution_summary()

    return results, sensors_thp_dict

def save_to_csv(det_times, min_det_time, scenario_characteristics, sensor_characteristics, signal, det_time_stats):
    os.makedirs(SIMULATION_CONFIG.output_folder, exist_ok=True)    
    det_times.to_csv(SIMULATION_CONFIG.output_folder + '/det_times.csv')
    det_time_stats.to_csv(SIMULATION_CONFIG.output_folder + '/det_time_stats.csv')
    min_det_time.to_csv(SIMULATION_CONFIG.output_folder + '/min_det_time.csv')
    scenario_characteristics.to_csv(SIMULATION_CONFIG.output_folder + '/scenario_characteristics.csv')
    sensor_characteristics.to_csv(SIMULATION_CONFIG.output_folder + '/sensor_characteristics.csv')
    signal.to_csv(SIMULATION_CONFIG.output_folder + '/signal.csv')
