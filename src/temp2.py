pickle_path = '/Users/kamilzawitaj/Documents/Studia/PW OKNO - AiR/Praca Mgr/code/leak_simulation/output_folder/pickle/'
import pickle

with open(pickle_path + 'scenario_metadata.pkl', 'rb') as file:
    scenario_metadata = pickle.load(file)

print(scenario_metadata)


from src.get_3sigma_threshold import get_1sigma_threshold
from src.config import SIMULATION_CONFIG

thresholds_series = get_1sigma_threshold()

wn = SIMULATION_CONFIG.create_network_real()
sensor_names = wn.junction_name_list
for location in sensor_names:
    print('location: ', location, '\tthresholds_series: ', thresholds_series[location])
