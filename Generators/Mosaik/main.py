import math
import mosaik_api
import mosaik
import numpy as np
from datetime import datetime
import os
import sys
import pandas as pd

META = {
    'api_version': '3.0',
    'type': 'time-based',
    'models': {
        'PowerPlant': {
            'public': True,
            'params': [],
            'attrs': [
                'voltage_a', 'voltage_b', 'voltage_c',
                'voltage_ab', 'voltage_bc', 'voltage_ca',
                'current_a', 'current_b', 'current_c',
                'frequency',
                'power_apparent', 'power_real', 'power_reactive',
                'power_factor'
            ],
        },
    },
}

class PowerPlantSim(mosaik_api.Simulator):
    def __init__(self, add_noise=False):
        super().__init__(META)
        self.eid_prefix = 'PowerPlant_'
        self.entities = {}
        self.time = 0
        self.step_size = None
        self._count = 0
        self.add_noise = add_noise
        # Add new default parameters
        self.base_voltage = 245.0
        self.base_current = 20.0
        self.base_frequency = 50.0

    def init(self, sid, time_resolution=1, step_size=1, add_noise=False, 
             base_voltage=245.0, base_current=20.0, base_frequency=50.0):
        self.time_resolution = time_resolution
        self.step_size = step_size
        self.add_noise = add_noise
        self.base_voltage = base_voltage
        self.base_current = base_current
        self.base_frequency = base_frequency
        return self.meta

    def create(self, num, model_params=None):
        if model_params is None:
            model_params = {}

        entities = []
        for i in range(num):
            eid = f"{self.eid_prefix}{self._count}"
            self.entities[eid] = {attr: 0.0 for attr in META['models']['PowerPlant']['attrs']}
            entities.append({'eid': eid, 'type': 'PowerPlant'})
            self._count += 1

        return entities

    def _calculate_variations(self, t):
        daily_var = np.sin(2 * np.pi * t / (24 * 3600)) * 0.1
        hourly_var = np.sin(2 * np.pi * t / 3600) * 0.05
        fast_var = np.sin(2 * np.pi * t / 300) * 0.02
        if self.add_noise:
            noise = np.random.normal(0, 0.01)
            return 1.0 + daily_var + hourly_var + fast_var + noise
        return 1.0 + daily_var + hourly_var + fast_var

    def _calculate_frequency(self, t, load_variation):
        freq_base = np.sin(2 * np.pi * t / 3600) * 0.02
        freq_load = -0.01 * (load_variation - 1)
        if self.add_noise:
            freq_noise = np.random.normal(0, 0.005)
            return self.base_frequency + freq_base + freq_load + freq_noise
        return self.base_frequency + freq_base + freq_load

    def step(self, time, inputs, max_advance):
        self.time = time
        omega = 2 * math.pi * self.base_frequency
        load_variation = self._calculate_variations(time)
        voltage_variation = 1.0 + np.sin(2 * np.pi * time / 600) * 0.01

        for eid, attrs in self.entities.items():
            if self.add_noise:
                angle_1 = 0 + np.random.normal(0, 0.5)
                angle_2 = -120 + np.random.normal(0, 0.5)
                angle_3 = 120 + np.random.normal(0, 0.5)
            else:
                angle_1 = 0
                angle_2 = -120
                angle_3 = 120
            
            if self.add_noise:
                v1 = self.base_voltage * voltage_variation * (1 + np.random.normal(0, 0.005))
                v2 = self.base_voltage * voltage_variation * (1 + np.random.normal(0, 0.005))
                v3 = self.base_voltage * voltage_variation * (1 + np.random.normal(0, 0.005))
            else:
                v1 = self.base_voltage * voltage_variation
                v2 = self.base_voltage * voltage_variation
                v3 = self.base_voltage * voltage_variation
            
            attrs['voltage_ab'] = np.sqrt(v1**2 + v2**2 - 2 * v1 * v2 * np.cos(np.radians(angle_1 - angle_2)))
            attrs['voltage_bc'] = np.sqrt(v2**2 + v3**2 - 2 * v2 * v3 * np.cos(np.radians(angle_2 - angle_3)))
            attrs['voltage_ca'] = np.sqrt(v3**2 + v1**2 - 2 * v3 * v1 * np.cos(np.radians(angle_3 - angle_1)))
            
            attrs['voltage_a'] = v1
            attrs['voltage_b'] = v2
            attrs['voltage_c'] = v3
            
            base_current = self.base_current * load_variation
            if self.add_noise:
                attrs['current_a'] = base_current * (1 + np.random.normal(0, 0.05))
                attrs['current_b'] = base_current * (1 + np.random.normal(0, 0.05))
                attrs['current_c'] = base_current * (1 + np.random.normal(0, 0.05))
            else:
                attrs['current_a'] = base_current
                attrs['current_b'] = base_current
                attrs['current_c'] = base_current
            
            base_pf = 0.95
            pf_load_impact = -0.05 * (load_variation - 1)
            if self.add_noise:
                attrs['power_factor'] = min(0.98, max(0.85, base_pf + pf_load_impact + np.random.normal(0, 0.01)))
            else:
                attrs['power_factor'] = min(0.98, max(0.85, base_pf + pf_load_impact))
            
            # Calculate powers for balanced three-phase system
            apparent_power = np.sqrt(3) * self.base_voltage * self.base_current  # S = √3 * V * I
            attrs['power_apparent'] = apparent_power
            attrs['power_real'] = apparent_power * attrs['power_factor']  # P = S * cos(φ)
            attrs['power_reactive'] = apparent_power * np.sin(np.arccos(attrs['power_factor']))  # Q = S * sin(φ)
            
            attrs['frequency'] = self._calculate_frequency(time, load_variation)

        return time + self.step_size

    def get_data(self, outputs):
        data = {}
        for eid, attrs_requested in outputs.items():
            data[eid] = {}
            for attr in attrs_requested:
                data[eid][attr] = self.entities[eid][attr]
        return data

def preprocess_data(input_file, output_file='processed.csv'):
    """Preprocess the simulation output data."""
    df = pd.read_csv(input_file)
    
    df['timestamp'] = pd.to_datetime(df['date'])
    processed = pd.DataFrame()
    processed['timestamp'] = pd.to_datetime(df['date'])
        
    # Map original columns to new names
    voltage_map = {
        'PowerPlantSim-0.PowerPlant_0-voltage_a': 'V1',
        'PowerPlantSim-0.PowerPlant_0-voltage_b': 'V2',
        'PowerPlantSim-0.PowerPlant_0-voltage_c': 'V3'
    }
    
    current_map = {
        'PowerPlantSim-0.PowerPlant_0-current_a': 'C1',
        'PowerPlantSim-0.PowerPlant_0-current_b': 'C2',
        'PowerPlantSim-0.PowerPlant_0-current_c': 'C3'
    }
    
    power_map = {
        'PowerPlantSim-0.PowerPlant_0-power_real': 'power_real',
        'PowerPlantSim-0.PowerPlant_0-power_reactive': 'power_effective',
        'PowerPlantSim-0.PowerPlant_0-power_apparent': 'power_apparent'
    }
    
    processed['frequency'] = df['PowerPlantSim-0.PowerPlant_0-frequency']
    
    for old_name, new_name in voltage_map.items():
        processed[new_name] = df[old_name]
        
    for old_name, new_name in current_map.items():
        processed[new_name] = df[old_name]
        
    processed['V1_V2'] = processed['V1'] - processed['V2']
    processed['V2_V3'] = processed['V2'] - processed['V3']
    processed['V1_V3'] = processed['V1'] - processed['V3']
    
    for old_name, new_name in power_map.items():
        processed[new_name] = df[old_name]
    
    processed.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")

def run_simulation(duration, with_noise=False, raw_output='results.csv', 
                  processed_output='processed.csv', preprocess=True,
                  base_voltage=245.0, base_current=20.0, base_frequency=50.0):
    """Run the power plant simulation and optionally preprocess the results."""
    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
    
    SIM_CONFIG = {
        'PowerPlantSim': {
            'python': '__main__:PowerPlantSim',
        },
        'CSVWriterSim': {
            'python': 'mosaik_csv_writer:CSVWriter',
        },
    }

    world = mosaik.World(SIM_CONFIG)
    power_plant = world.start('PowerPlantSim', 
                            step_size=1, 
                            add_noise=with_noise,
                            base_voltage=base_voltage,
                            base_current=base_current,
                            base_frequency=base_frequency)
    pp_entities = power_plant.PowerPlant.create(1)
    
    csv_writer = world.start(
        'CSVWriterSim',
        raw_output,
        start_date=formatted_time,
        time_resolution=1
    )
    csv_entities = csv_writer.CSVWriter.create(1)

    world.connect(pp_entities[0], csv_entities[0], 
                 'voltage_a', 'voltage_b', 'voltage_c',
                 'voltage_ab', 'voltage_bc', 'voltage_ca',
                 'current_a', 'current_b', 'current_c',
                 'frequency', 'power_apparent', 'power_real',
                 'power_reactive', 'power_factor')

    print(f"Starting simulation {'with' if with_noise else 'without'} noise...")
    print(f"Using base voltage={base_voltage}V, current={base_current}A, frequency={base_frequency}Hz")
    world.run(until=duration)
    print(f"Simulation completed. Raw results saved to {raw_output}")
    
    if preprocess:
        print("Starting data preprocessing...")
        preprocess_data(raw_output, processed_output)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run power plant simulation and data processing')
    parser.add_argument('duration', type=int, help='Simulation duration in seconds')
    parser.add_argument('--no-noise', action='store_true', help='Disable noise in simulation')
    parser.add_argument('--processed-output', default='Mosaik.csv', help='Processed output file')
    parser.add_argument('--voltage', type=float, default=245.0, help='Base voltage value (default: 245.0)')
    parser.add_argument('--current', type=float, default=20.0, help='Base current value (default: 20.0)')
    parser.add_argument('--frequency', type=float, default=50.0, help='Base frequency value (default: 50.0)')
    
    args = parser.parse_args()
    
    if args.duration <= 0:
        print("Error: Duration must be positive")
        sys.exit(1)
        
    try:
        run_simulation(
            duration=args.duration,
            with_noise=not args.no_noise,
            raw_output='results.csv',
            processed_output=args.processed_output,
            base_voltage=args.voltage,
            base_current=args.current,
            base_frequency=args.frequency
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)