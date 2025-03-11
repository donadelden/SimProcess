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
    def __init__(self, noise_config=None):
        super().__init__(META)
        self.eid_prefix = 'PowerPlant_'
        self.entities = {}
        self.time = 0
        self.step_size = None
        self._count = 0
        
        # Default noise configuration
        self.noise_config = noise_config or {
            'type': 'gaussian',  # default noise type
            'scale': 0.01,       # default scale
            'impulse_prob': 0.01,  # for impulse noise
            'poisson_lambda': 2.0   # for poisson noise
        }
        
        self.base_voltage = 245.0
        self.base_current = 20.0
        self.base_frequency = 50.0

    def init(self, sid, time_resolution=1, step_size=1, noise_config=None, 
             base_voltage=245.0, base_current=20.0, base_frequency=50.0):
        self.time_resolution = time_resolution
        self.step_size = step_size
        if noise_config:
            self.noise_config = noise_config
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

    def add_noise(self, base_value, noise_type=None, scale=None, **kwargs):
        """
        Add noise to a base value using specified noise model.
        
        Parameters:
        -----------
        base_value : float
            Original value to add noise to
        noise_type : str, optional
            Type of noise to apply (uses config if None)
        scale : float, optional
            Scale of noise (uses config if None)
        **kwargs : dict
            Additional parameters for specific noise types
        """
        # Use config values if not specified
        if noise_type is None:
            noise_type = self.noise_config.get('type', 'gaussian')
        if scale is None:
            scale = self.noise_config.get('scale', 0.01)
            
        # Get additional parameters from noise_config if not in kwargs
        for key, value in self.noise_config.items():
            if key not in kwargs and key not in ['type', 'scale']:
                kwargs[key] = value
                
        # Apply the selected noise type
        if noise_type == 'gaussian':
            return base_value * (1 + np.random.normal(0, scale))
            
        elif noise_type == 'uniform':
            return base_value * (1 + np.random.uniform(-scale, scale))
            
        elif noise_type == 'laplace':
            return base_value * (1 + np.random.laplace(0, scale))
            
        elif noise_type == 'poisson':
            lambda_param = kwargs.get('poisson_lambda', 2.0)
            return base_value + np.random.poisson(lam=lambda_param) * scale
            
        elif noise_type == 'impulse':
            impulse_prob = kwargs.get('impulse_prob', 0.01)
            if np.random.rand() < impulse_prob:
                return base_value + np.random.choice([-0.1, 0.1]) * base_value
            return base_value
            
        elif noise_type == 'brownian':
            return base_value + np.cumsum(np.random.normal(0, scale, size=1000))[-1]
            
        elif noise_type == 'pink':
            # Simple approximation of pink noise using multiple octaves of noise
            noise = 0
            for i in range(1, 5):
                noise += np.random.normal(0, scale / i)
            return base_value * (1 + noise)
            
        elif noise_type == 'none':
            return base_value
            
        else:
            # Default to no noise for unknown types
            return base_value
    
    def step(self, time, inputs, max_advance):
        self.time = time
        
        # Get base voltage variation (applies to all phases)
        voltage_variation = 1.0
        if self.noise_config['type'] != 'none':
            voltage_variation = self.add_noise(1.0)
        
        for eid, attrs in self.entities.items():
            # Apply phase angle variations
            if self.noise_config['type'] != 'none':
                angle_1 = 0 + np.random.normal(0, 0.1)
                angle_2 = -120 + np.random.normal(0, 0.1)
                angle_3 = 120 + np.random.normal(0, 0.1)
            else:
                angle_1 = 0
                angle_2 = -120
                angle_3 = 120
            
            # Apply voltage noise to each phase
            if self.noise_config['type'] != 'none':
                v1 = self.add_noise(self.base_voltage * voltage_variation)
                v2 = self.add_noise(self.base_voltage * voltage_variation)
                v3 = self.add_noise(self.base_voltage * voltage_variation)
            else:
                v1 = self.base_voltage * voltage_variation
                v2 = self.base_voltage * voltage_variation
                v3 = self.base_voltage * voltage_variation
            
            # Set phase voltages
            attrs['voltage_a'] = v1
            attrs['voltage_b'] = v2
            attrs['voltage_c'] = v3
            
            # Calculate line-to-line voltages
            attrs['voltage_ab'] = np.sqrt(v1**2 + v2**2 - 2 * v1 * v2 * np.cos(np.radians(angle_1 - angle_2)))
            attrs['voltage_bc'] = np.sqrt(v2**2 + v3**2 - 2 * v2 * v3 * np.cos(np.radians(angle_2 - angle_3)))
            attrs['voltage_ca'] = np.sqrt(v3**2 + v1**2 - 2 * v3 * v1 * np.cos(np.radians(angle_3 - angle_1)))
            
            # Apply current noise to each phase
            base_current = self.base_current
            if self.noise_config['type'] != 'none':
                attrs['current_a'] = self.add_noise(base_current, scale=0.05)
                attrs['current_b'] = self.add_noise(base_current, scale=0.05)
                attrs['current_c'] = self.add_noise(base_current, scale=0.05)
            else:
                attrs['current_a'] = base_current
                attrs['current_b'] = base_current
                attrs['current_c'] = base_current
            
            # Apply power factor noise
            base_pf = 0.95
            pf_load_impact = -0.05
            if self.noise_config['type'] != 'none':
                pf_noise = np.random.normal(0, 0.01)
                attrs['power_factor'] = min(0.98, max(0.85, base_pf + pf_load_impact + pf_noise))
            else:
                attrs['power_factor'] = min(0.98, max(0.85, base_pf + pf_load_impact))
            
            # Calculate powers for balanced three-phase system
            actual_voltage_avg = (attrs['voltage_a'] + attrs['voltage_b'] + attrs['voltage_c']) / 3
            actual_current_avg = (attrs['current_a'] + attrs['current_b'] + attrs['current_c']) / 3
            apparent_power = np.sqrt(3) * actual_voltage_avg * actual_current_avg  # S = √3 * V * I
            attrs['power_apparent'] = apparent_power
            attrs['power_real'] = apparent_power * attrs['power_factor']  # P = S * cos(φ)
            attrs['power_reactive'] = apparent_power * np.sin(np.arccos(attrs['power_factor']))  # Q = S * sin(φ)
            
            # Apply frequency noise
            if self.noise_config['type'] != 'none':
                attrs['frequency'] = self.add_noise(self.base_frequency, scale=0.005)
            else:
                attrs['frequency'] = self.base_frequency

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


def run_simulation(duration, noise_config=None, raw_output='results.csv', 
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

    # Print information about the simulation
    print(f"Starting simulation with {noise_config['type']} noise (scale={noise_config['scale']})...")
    print(f"Using base voltage={base_voltage}V, current={base_current}A, frequency={base_frequency}Hz")

    world = mosaik.World(SIM_CONFIG)
    power_plant = world.start('PowerPlantSim', 
                             step_size=1, 
                             noise_config=noise_config,
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

    world.run(until=duration)
    print(f"Simulation completed. Raw results saved to {raw_output}")
    
    if preprocess:
        print("Starting data preprocessing...")
        preprocess_data(raw_output, processed_output)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run power plant simulation with various noise models')
    parser.add_argument('duration', type=int, help='Simulation duration in seconds')
    
    # Basic parameters
    parser.add_argument('--voltage', type=float, default=245.0, help='Base voltage value (default: 245.0)')
    parser.add_argument('--current', type=float, default=20.0, help='Base current value (default: 20.0)')
    parser.add_argument('--frequency', type=float, default=50.0, help='Base frequency value (default: 50.0)')
    parser.add_argument('--processed-output', default='Mosaik.csv', help='Processed output file')
    
    # Noise type selection
    parser.add_argument('--noise-type', type=str, default='gaussian', 
                        choices=['gaussian', 'uniform', 'laplace', 'poisson', 
                                 'impulse', 'brownian', 'pink', 'none'],
                        help='Type of noise to apply (default: gaussian)')
    
    # Noise scale (intensity)
    parser.add_argument('--noise-scale', type=float, default=0.01,
                        help='Scale/intensity of the noise (default: 0.01)')
    
    # Additional noise parameters
    parser.add_argument('--impulse-prob', type=float, default=0.01,
                        help='Probability of impulse noise occurrence (default: 0.01)')
    parser.add_argument('--poisson-lambda', type=float, default=2.0,
                        help='Lambda parameter for Poisson noise (default: 2.0)')
    
    args = parser.parse_args()
    
    if args.duration <= 0:
        print("Error: Duration must be positive")
        sys.exit(1)
    
    # Configure noise parameters based on command line arguments
    noise_config = {
        'type': args.noise_type,
        'scale': args.noise_scale,
        'impulse_prob': args.impulse_prob,
        'poisson_lambda': args.poisson_lambda
    }
    
    try:
        run_simulation(
            duration=args.duration,
            noise_config=noise_config,
            raw_output='results.csv',
            processed_output=args.processed_output,
            base_voltage=args.voltage,
            base_current=args.current,
            base_frequency=args.frequency
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)