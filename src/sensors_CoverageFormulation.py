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


def get_sensor_locations(wn, signal, threshold_parameters, sensor_budget):

    #scenario_names = signal.columns.tolist()[2:]
    scenario_names = [col for col in signal.columns if col not in ['T', 'Node']]
    sensor_names = wn.junction_name_list
    sample_times = np.arange(0, wn.options.time.duration, wn.options.time.hydraulic_timestep)

    thresholds_series = get_1sigma_threshold()

    sensors = {}
    sensors_thp_dict = {}
    grouped_sensors_list = []
    cost_data_list = []

    for location in sensor_names:
        position = chama.sensors.Stationary(location)
        same_location_sensors = []
        for threshold_parameter in threshold_parameters:
            sensor_name = f'Node{location}_thp{threshold_parameter}'
            sensors_thp_dict[sensor_name] = threshold_parameter
            detector = chama.sensors.Point(threshold_parameter * thresholds_series[location], sample_times)
            stationary_pt_sensor = chama.sensors.Sensor(position, detector)
            sensors[sensor_name] = stationary_pt_sensor
            same_location_sensors.append(sensor_name)
            cost_data_list.append({'Sensor': sensor_name, 'Cost': 1})
        grouped_sensors_list.append(same_location_sensors)

    det_times = chama.impact.extract_detection_times(signal, sensors)
    det_time_stats = chama.impact.detection_time_stats(det_times)
    min_det_time = det_time_stats[['Scenario','Sensor','Min']].copy()
    min_det_time.rename(columns = {'Min':'Impact'}, inplace = True)
    min_det_time = min_det_time.dropna(subset=['Impact'])

    scenario_characteristics = pd.DataFrame({'Scenario': scenario_names,
                                             'Undetected Impact': sample_times.max()*1.5})
    #sensor_characteristics = pd.DataFrame({'Sensor': sensor_names,'Cost': 1})
    sensor_characteristics = pd.DataFrame(cost_data_list)

    coverage_df = min_det_time.groupby('Sensor')['Scenario'].apply(list).reset_index()
    coverage_df.rename(columns={'Scenario': 'Coverage'}, inplace=True)
    entity_df = scenario_characteristics.copy()
    entity_df.rename(columns={'Scenario': 'Entity'}, inplace=True)

    results = {}

    coverage_percent_list = []


    coverageform = chama.optimize.CoverageFormulation()
    model = coverageform.create_pyomo_model(coverage=coverage_df,
                                          sensor=sensor_characteristics,
                                          entity=entity_df)

    valid_sensors = set(coverage_df['Sensor'])

    for sensor_group in grouped_sensors_list:
        filtered_group = [s for s in sensor_group if s in valid_sensors]
        if filtered_group:
            coverageform.add_grouping_constraint(filtered_group, max_select=1)
        #coverageform.add_grouping_constraint(sensor_group, max_select=1)


    coverageform.solve_pyomo_model(sensor_budget=sensor_budget)
    results = coverageform.create_solution_summary()

    return results, sensors_thp_dict

def save_to_csv(det_times, min_det_time, scenario_characteristics, sensor_characteristics, signal, det_time_stats):
    os.makedirs(SIMULATION_CONFIG.output_folder, exist_ok=True)    
    det_times.to_csv(SIMULATION_CONFIG.output_folder + '/det_times.csv')
    det_time_stats.to_csv(SIMULATION_CONFIG.output_folder + '/det_time_stats.csv')
    min_det_time.to_csv(SIMULATION_CONFIG.output_folder + '/min_det_time.csv')
    scenario_characteristics.to_csv(SIMULATION_CONFIG.output_folder + '/scenario_characteristics.csv')
    sensor_characteristics.to_csv(SIMULATION_CONFIG.output_folder + '/sensor_characteristics.csv')
    signal.to_csv(SIMULATION_CONFIG.output_folder + '/signal.csv')
