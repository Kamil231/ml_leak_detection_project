import wntr
from dataclasses import dataclass, field
from typing import List
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / 'output_folder'
OUTPUT_DIR.mkdir(exist_ok=True)



@dataclass
class TimeParameters:
    duration_s: int
    hydraulic_timestep_s: int
    report_timestep_s: int
    leak_duration: float
    
@dataclass
class WaterNetworkParameters:
    inp_file_path: str
    required_pressure_m: float
    minimum_pressure_m: float
    pipe_diameter_m: float
    demand_model: str

@dataclass
class ScenariosParameters:
    demand_noise_parameter: float
    sensor_noise_parameter: float
    sigma3_sim_number: int
    sensor_budget: List[int] = field(default_factory=list)
    
@dataclass
class SimulationConfig:
    output_folder: str
    time: TimeParameters
    network: WaterNetworkParameters
    scenarios: ScenariosParameters
    
    def create_newtork(self)-> wntr.network.WaterNetworkModel:
        
        wn = wntr.network.WaterNetworkModel(self.network.inp_file_path)
        wn.options.time.duration = self.time.duration_s
        wn.options.time.hydraulic_timestep = self.time.hydraulic_timestep_s
        wn.options.time.report_timestep = self.time.report_timestep_s
        wn.options.hydraulic.required_pressure = self.network.required_pressure_m
        wn.options.hydraulic.minimum_pressure = self.network.minimum_pressure_m
        wn.options.hydraulic.demand_model = self.network.demand_model  
        
        return wn

SIMULATION_CONFIG = SimulationConfig(
    #output_folder = 'output_folder',
    output_folder = OUTPUT_DIR,
    time=TimeParameters(
        duration_s = 7 * 24 * 3600, #48 * 3600, #0.5 * 96 * 3600,
        hydraulic_timestep_s = 3600, #1800, #300, #1800/600,
        report_timestep_s = 3600, #300, #1800,
        leak_duration = 0 #3.0 # if leak_duration = 0, leak is presesnt until the end of simulation
    ),
    network=WaterNetworkParameters(
        inp_file_path = "data/Net3.inp",
        required_pressure_m = 15.0,
        minimum_pressure_m = 0,
        pipe_diameter_m = 0.9144,
        demand_model = "PDD"
    ),
    scenarios=ScenariosParameters(
        demand_noise_parameter = .005,
        sensor_noise_parameter = .033,
        sigma3_sim_number = 30,
        #sensor_budget = list(range(1,16)) # + list(range(10, 160, 1))
        sensor_budget = list(range(1,11)) # + list(range(10, 160, 1))
    )
)