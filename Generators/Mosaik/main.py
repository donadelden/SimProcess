import math
import mosaik_api
import mosaik
import numpy as np
from datetime import datetime
import os
import sys
import pandas as pd
from sklearn.mixture import GaussianMixture

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
    def __init__(self, noise_configs=None):
        super().__init__(META)
        self.eid_prefix = 'PowerPlant_'
        self.entities = {}
        self.time = 0
        self.step_size = None
        self._count = 0
        
        # Default noise configuration - now a list to support multiple noise types
        self.noise_configs = noise_configs or [{
            'type': 'gaussian',  # default noise type
            'scale': 0.01,       # default scale
            'impulse_prob': 0.01,  # for impulse noise
            'poisson_lambda': 2.0   # for poisson noise
        }]
        
        self.base_voltage = 245.0
        self.base_current = 20.0
        self.base_frequency = 50.0
        
        # For fitted GMM models
        self.gmm_models = {}
        
        # Initialize GMM models if provided in config
        for noise_config in self.noise_configs:
            if noise_config.get('type') == 'gmm' and noise_config.get('reference_file'):
                self.init_gmm_models(noise_config)

    def init_gmm_models(self, noise_config):
        """Initialize GMM models from reference data."""
        reference_file = noise_config.get('reference_file')
        n_components = int(noise_config.get('n_components', 2))
        
        try:
            # Read the reference data
            print(f"Loading reference data from {reference_file}...")
            ref_df = pd.read_csv(reference_file)
            
            # Columns to process - exclude timestamp
            columns_to_process = [col for col in ref_df.columns if col != 'timestamp']
            
            # Create GMM models for each column
            for column in columns_to_process:
                # Extract data
                data = ref_df[column].values
                
                # Skip if column contains non-numeric values
                if not np.issubdtype(data.dtype, np.number):
                    print(f"Skipping non-numeric column: {column}")
                    continue
                
                # Normalize the data for better GMM fitting
                data_mean = np.mean(data)
                data_std = np.std(data)
                if data_std == 0:
                    print(f"Skipping column with zero standard deviation: {column}")
                    continue
                    
                normalized_data = (data - data_mean) / data_std
                
                # Fit GMM to the data
                gmm = GaussianMixture(n_components=n_components, random_state=42)
                gmm.fit(normalized_data.reshape(-1, 1))
                
                # Store GMM model and normalization parameters
                self.gmm_models[column] = {
                    'gmm': gmm,
                    'mean': data_mean,
                    'std': data_std,
                    'means': gmm.means_.flatten().tolist(),
                    'variances': gmm.covariances_.flatten().tolist(),
                    'weights': gmm.weights_.tolist()
                }
                
            print(f"Successfully fitted GMM models for {len(self.gmm_models)} columns")
            
        except Exception as e:
            print(f"Error initializing GMM models: {e}")

    def init(self, sid, time_resolution=1, step_size=1, noise_configs=None, 
             base_voltage=245.0, base_current=20.0, base_frequency=50.0):
        self.time_resolution = time_resolution
        self.step_size = step_size
        if noise_configs:
            self.noise_configs = noise_configs
            # Reinitialize GMM models if needed
            self.gmm_models = {}
            for noise_config in self.noise_configs:
                if noise_config.get('type') == 'gmm' and noise_config.get('reference_file'):
                    self.init_gmm_models(noise_config)
                    
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

    def add_single_noise(self, base_value, noise_config, attr_name=None):
        """
        Add a single type of noise to a base value.
        
        Parameters:
        -----------
        base_value : float
            Original value to add noise to
        noise_config : dict
            Configuration for the noise to apply
        attr_name : str, optional
            Name of the attribute being processed (for GMM model selection)
        """
        noise_type = noise_config.get('type', 'gaussian')
        scale = noise_config.get('scale', 0.01)
            
        # Apply the selected noise type
        if noise_type == 'gaussian':
            return base_value * (1 + np.random.normal(0, scale))
            
        elif noise_type == 'uniform':
            return base_value * (1 + np.random.uniform(-scale, scale))
            
        elif noise_type == 'laplace':
            return base_value * (1 + np.random.laplace(0, scale))
            
        elif noise_type == 'poisson':
            lambda_param = noise_config.get('poisson_lambda', 2.0)
            return base_value + np.random.poisson(lam=lambda_param) * scale
            
        elif noise_type == 'impulse':
            impulse_prob = noise_config.get('impulse_prob', 0.01)
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
            
        elif noise_type == 'gmm':
            # Check if we have a fitted GMM model for this attribute
            if attr_name and attr_name in self.gmm_models:
                model_info = self.gmm_models[attr_name]
                gmm = model_info['gmm']
                
                # Generate noise from the fitted GMM
                # First select a component based on weights
                component = np.random.choice(len(model_info['weights']), p=model_info['weights'])
                
                # Then generate noise from that component
                noise_value = np.random.normal(
                    model_info['means'][component], 
                    np.sqrt(model_info['variances'][component])
                )
                
                # Denormalize and apply the noise
                scaled_noise = noise_value * model_info['std'] * scale
                return base_value + scaled_noise
                
            else:
                # Use manual GMM parameters if no fitted model exists
                n_components = int(noise_config.get('n_components', 2))
                gmm_means = noise_config.get('means', [0.0, 0.0])
                gmm_variances = noise_config.get('variances', [0.01, 0.05])
                gmm_weights = noise_config.get('weights', [0.7, 0.3])
                
                # Ensure parameters are in the correct format
                if not isinstance(gmm_means, list):
                    gmm_means = [float(x) for x in str(gmm_means).replace('[', '').replace(']', '').split(',')]
                if not isinstance(gmm_variances, list):
                    gmm_variances = [float(x) for x in str(gmm_variances).replace('[', '').replace(']', '').split(',')]
                if not isinstance(gmm_weights, list):
                    gmm_weights = [float(x) for x in str(gmm_weights).replace('[', '').replace(']', '').split(',')]
                    
                # Ensure weights sum to 1
                gmm_weights = np.array(gmm_weights)
                gmm_weights = gmm_weights / gmm_weights.sum()
                
                # Select component based on weights
                component = np.random.choice(len(gmm_weights), p=gmm_weights)
                
                # Generate noise from selected component
                noise_value = np.random.normal(gmm_means[component], np.sqrt(gmm_variances[component]))
                
                return base_value * (1 + noise_value * scale)
            
        elif noise_type == 'none':
            return base_value
            
        else:
            # Default to no noise for unknown types
            return base_value
    
    def add_noise(self, base_value, attr_name=None):
        """
        Apply multiple noise types sequentially to a base value.
        
        Parameters:
        -----------
        base_value : float
            Original value to add noise to
        attr_name : str, optional
            Name of the attribute being processed (for GMM model selection)
        """
        current_value = base_value
        
        # Apply each noise type in sequence
        for noise_config in self.noise_configs:
            if noise_config['type'] != 'none':
                current_value = self.add_single_noise(current_value, noise_config, attr_name)
        
        return current_value
    
    def step(self, time, inputs, max_advance):
        self.time = time
        
        # Check if any noise is being applied
        has_noise = any(config['type'] != 'none' for config in self.noise_configs)
        
        # Get base voltage variation (applies to all phases)
        voltage_variation = 1.0
        if has_noise:
            voltage_variation = self.add_noise(1.0, 'voltage_variation')
        
        for eid, attrs in self.entities.items():
            # Apply phase angle variations
            if has_noise:
                angle_1 = 0 + np.random.normal(0, 0.1)
                angle_2 = -120 + np.random.normal(0, 0.1)
                angle_3 = 120 + np.random.normal(0, 0.1)
            else:
                angle_1 = 0
                angle_2 = -120
                angle_3 = 120
            
            # Apply voltage noise to each phase
            if has_noise:
                v1 = self.add_noise(self.base_voltage * voltage_variation, 'voltage_a')
                v2 = self.add_noise(self.base_voltage * voltage_variation, 'voltage_b')
                v3 = self.add_noise(self.base_voltage * voltage_variation, 'voltage_c')
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
            if has_noise:
                attrs['current_a'] = self.add_noise(base_current, 'current_a')
                attrs['current_b'] = self.add_noise(base_current, 'current_b')
                attrs['current_c'] = self.add_noise(base_current, 'current_c')
            else:
                attrs['current_a'] = base_current
                attrs['current_b'] = base_current
                attrs['current_c'] = base_current
            
            # Apply power factor noise
            base_pf = 0.95
            pf_load_impact = -0.05
            if has_noise:
                pf_noise = np.random.normal(0, 0.01)
                attrs['power_factor'] = min(0.98, max(0.85, base_pf + pf_load_impact + pf_noise))
            else:
                attrs['power_factor'] = min(0.98, max(0.85, base_pf + pf_load_impact))
            
            # Calculate powers for balanced three-phase system
            actual_voltage_avg = (attrs['voltage_a'] + attrs['voltage_b'] + attrs['voltage_c']) / 3
            actual_current_avg = (attrs['current_a'] + attrs['current_b'] + attrs['current_c']) / 3
            apparent_power = np.sqrt(3) * actual_voltage_avg * actual_current_avg  # S = √3 * V * I
            attrs['power_apparent'] = apparent_power
            attrs['power_real'] = self.add_noise(apparent_power * attrs['power_factor'], 'power_real')  # P = S * cos(φ)
            attrs['power_reactive'] = self.add_noise(apparent_power * np.sin(np.arccos(attrs['power_factor'])), 'power_reactive')  # Q = S * sin(φ)
            
            # Apply frequency noise
            if has_noise:
                attrs['frequency'] = self.add_noise(self.base_frequency, 'frequency')
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


def run_simulation(duration, noise_configs=None, raw_output='results.csv', 
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
    noise_descriptions = []
    for config in noise_configs:
        noise_descriptions.append(f"{config['type']} (scale={config['scale']})")
    
    print(f"Starting simulation with noise types: {', '.join(noise_descriptions)}")
    print(f"Using base voltage={base_voltage}V, current={base_current}A, frequency={base_frequency}Hz")

    world = mosaik.World(SIM_CONFIG)
    power_plant = world.start('PowerPlantSim', 
                             step_size=1, 
                             noise_configs=noise_configs,
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


def parse_noise_arg(arg_string):
    """Parse a noise specification string into a dictionary."""
    parts = arg_string.split(':')
    noise_type = parts[0]
    
    config = {
        'type': noise_type,
        'scale': 0.01,  # default values
        'impulse_prob': 0.01,
        'poisson_lambda': 2.0
    }
    
    # Parse additional parameters if provided
    if len(parts) > 1:
        # First join all parts except the first one back together
        # This handles cases where ':' might appear in the parameter values
        param_string = ':'.join(parts[1:])
        params = param_string.split(',')
        
        for param in params:
            if '=' in param:
                key, value = param.split('=', 1)  # Split only on first '='
                
                # Handle list values enclosed in square brackets
                if value.startswith('[') and ']' in value:
                    # Extract the list content
                    list_str = value.strip('[]')
                    # Split by comma and convert each value to float
                    try:
                        value_list = [float(x.strip()) for x in list_str.split(',')]
                        config[key] = value_list
                    except ValueError:
                        print(f"Warning: Could not parse list value '{value}', using default")
                else:
                    # Try to convert to float
                    try:
                        config[key] = float(value)
                    except ValueError:
                        # If conversion fails, keep as string
                        config[key] = value
    
    return config


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run power plant simulation with various noise models')
    parser.add_argument('duration', type=int, help='Simulation duration in seconds')
    
    # Basic parameters
    parser.add_argument('--voltage', type=float, default=245.0, help='Base voltage value (default: 245.0)')
    parser.add_argument('--current', type=float, default=20.0, help='Base current value (default: 20.0)')
    parser.add_argument('--frequency', type=float, default=50.0, help='Base frequency value (default: 50.0)')
    parser.add_argument('--processed-output', default='Mosaik.csv', help='Processed output file')
    
    # Multiple noise types using a more flexible format
    parser.add_argument('--noise', action='append', default=[], 
                       help='Noise specification in format "type:param1=value1,param2=value2". '
                            'Can be used multiple times for layered noise. '
                            'Available types: gaussian, uniform, laplace, poisson, impulse, brownian, pink, gmm, none')
    
    args = parser.parse_args()
    
    if args.duration <= 0:
        print("Error: Duration must be positive")
        sys.exit(1)
    
    # Parse noise configurations
    noise_configs = []
    if not args.noise:
        # Default to gaussian noise if none specified
        noise_configs.append({
            'type': 'gaussian',
            'scale': 0.01,
            'impulse_prob': 0.01,
            'poisson_lambda': 2.0
        })
    else:
        for noise_arg in args.noise:
            noise_configs.append(parse_noise_arg(noise_arg))
    
    try:
        run_simulation(
            duration=args.duration,
            noise_configs=noise_configs,
            raw_output='results.csv',
            processed_output=args.processed_output,
            base_voltage=args.voltage,
            base_current=args.current,
            base_frequency=args.frequency
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)