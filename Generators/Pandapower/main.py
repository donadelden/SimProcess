import pandapower as pp
import pandas as pd
import numpy as np
from numpy.random import normal
from datetime import datetime, timedelta
import sys
import traceback
import argparse
from sklearn.mixture import GaussianMixture

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Generate power network measurements data with various noise models'
    )
    parser.add_argument(
        'duration',
        type=int,
        help='Duration in seconds for which to generate data'
    )
    parser.add_argument(
        '--voltage',
        type=float,
        default=245.0,
        help='Base voltage value (default: 245.0)'
    )
    parser.add_argument(
        '--current',
        type=float,
        default=20.0,
        help='Base current value (default: 20.0)'
    )
    parser.add_argument(
        '--frequency',
        type=float,
        default=50.0,
        help='Base frequency value (default: 50.0)'
    )
    parser.add_argument(
        '--output-noisy',
        type=str,
        default='Panda.csv',
        help='Output file for noisy data (default: Panda.csv)'
    )
    parser.add_argument(
        '--output-noiseless',
        type=str,
        default='Panda_denoised.csv',
        help='Output file for noiseless data (default: Panda_denoised.csv)'
    )
    
    # Add multiple noise types using a more flexible format
    parser.add_argument(
        '--noise', 
        action='append', 
        default=[], 
        help='Noise specification in format "type:param1=value1,param2=value2". '
             'Can be used multiple times for layered noise. '
             'Available types: gaussian, uniform, laplace, poisson, impulse, brownian, pink, gmm, none'
    )
    
    # GMM specific parameters
    parser.add_argument(
        '--reference-file',
        type=str,
        default=None,
        help='Reference file for GMM model fitting'
    )
    parser.add_argument(
        '--gmm-components',
        type=int,
        default=2,
        help='Number of components for GMM model (default: 2)'
    )
    parser.add_argument(
        '--gmm-means',
        type=str,
        default='0.0,0.0',
        help='Comma-separated means for GMM model (default: "0.0,0.0")'
    )
    parser.add_argument(
        '--gmm-variances',
        type=str,
        default='0.01,0.05',
        help='Comma-separated variances for GMM model (default: "0.01,0.05")'
    )
    parser.add_argument(
        '--gmm-weights',
        type=str,
        default='0.7,0.3',
        help='Comma-separated weights for GMM model (default: "0.7,0.3")'
    )
    
    return parser.parse_args()

def create_simple_network():
    """Create a simplified network with one generator and two loads"""
    try:
        net = pp.create_empty_network()
        
        b1 = pp.create_bus(net, vn_kv=0.4, name="Generation Bus")
        b2 = pp.create_bus(net, vn_kv=0.4, name="Load Bus 1")
        b3 = pp.create_bus(net, vn_kv=0.4, name="Load Bus 2")
        
        pp.create_ext_grid(net, bus=b1, vm_pu=1.0, name="Grid Connection")
        pp.create_gen(net, bus=b1, p_mw=0.01, vm_pu=1.0, name="Generator")
        
        pp.create_load(net, bus=b2, p_mw=0.003, q_mvar=0.001, name="Load 1")
        pp.create_load(net, bus=b3, p_mw=0.004, q_mvar=0.001, name="Load 2")
        
        pp.create_line(net, from_bus=b1, to_bus=b2, length_km=0.1, std_type="NAYY 4x50 SE", name="Line 1-2")
        pp.create_line(net, from_bus=b1, to_bus=b3, length_km=0.1, std_type="NAYY 4x50 SE", name="Line 1-3")
        
        return net
        
    except Exception as e:
        print(f"Error creating network: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

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

class NoiseGenerator:
    def __init__(self, noise_configs=None):
        """
        Initialize the noise generator with the specified configurations.
        
        Parameters:
        -----------
        noise_configs : list of dict
            List of noise configuration dictionaries
        """
        # Default noise configuration - now a list to support multiple noise types
        self.noise_configs = noise_configs or [{
            'type': 'gaussian',  # default noise type
            'scale': 0.01,       # default scale
            'impulse_prob': 0.01,  # for impulse noise
            'poisson_lambda': 2.0   # for poisson noise
        }]
        
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
            traceback.print_exc()
    
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
    
    def has_noise(self):
        """Check if any noise is being applied."""
        return any(config['type'] != 'none' for config in self.noise_configs)


# Function to determine the current multiplier based on the simulation progress
def get_current_multiplier(current_index, total_duration):
    """
    Get the current multiplier based on the current index position in the simulation.
    
    Parameters:
    -----------
    current_index : int
        Current index in the simulation timeline
    total_duration : int
        Total duration of the simulation in steps
        
    Returns:
    --------
    float
        Multiplier to apply to the base current value
    """
    # Define breakpoints for the current pattern (as percentages)
    breakpoint_percentages = [0.2, 0.45, 0.65, 0.85, 1.0]
    breakpoints = [int(bp * total_duration) for bp in breakpoint_percentages]
    
    # Define current multipliers for each segment
    # 1.0 = base value, 1.2 = base+20%, 0.85 = base-15%
    multipliers = [1.0, 1.2, 1.0, 0.85, 1.0]
    
    # Find which segment of the pattern we're in
    for i, breakpoint in enumerate(breakpoints):
        if current_index < breakpoint:
            return multipliers[i]
    
    # Default to last segment
    return multipliers[-1]


def simulate_measurements(net, timestamp, noise_generator, freq=50.0, voltage=245.0, current=20.0, 
                         current_index=0, total_duration=None):
    """Simulate electrical measurements with configurable noise and current pattern"""
    try:
        pp.runpp(net, calculate_voltage_angles=True)
        
        measurements = {}
        base_frequency = freq
        has_noise = noise_generator.has_noise()
        
        bus = 0
        t = timestamp.timestamp()
        
        # Apply the current pattern if total_duration is provided
        current_multiplier = 1.0
        if total_duration is not None:
            current_multiplier = get_current_multiplier(current_index, total_duration)
        
        # Get base voltage variation (applies to all phases)
        voltage_variation = 1.0
        if has_noise:
            voltage_variation = noise_generator.add_noise(1.0, 'voltage_variation')
        
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
        base_voltage = voltage
        if has_noise:
            v1 = noise_generator.add_noise(base_voltage * voltage_variation, 'voltage_a')
            v2 = noise_generator.add_noise(base_voltage * voltage_variation, 'voltage_b')
            v3 = noise_generator.add_noise(base_voltage * voltage_variation, 'voltage_c')
        else:
            v1 = base_voltage * voltage_variation
            v2 = base_voltage * voltage_variation
            v3 = base_voltage * voltage_variation
            
        # Calculate line-to-line voltages using the more accurate formula from Mosaik
        v_ab = np.sqrt(v1**2 + v2**2 - 2 * v1 * v2 * np.cos(np.radians(angle_1 - angle_2)))
        v_bc = np.sqrt(v2**2 + v3**2 - 2 * v2 * v3 * np.cos(np.radians(angle_2 - angle_3)))
        v_ca = np.sqrt(v3**2 + v1**2 - 2 * v3 * v1 * np.cos(np.radians(angle_3 - angle_1)))
        
        # Apply the pattern multiplier to the base current before applying noise
        modified_base_current = current * current_multiplier
        
        # Apply current noise to each phase
        if has_noise:
            i1 = noise_generator.add_noise(modified_base_current, 'current_a')
            i2 = noise_generator.add_noise(modified_base_current, 'current_b')
            i3 = noise_generator.add_noise(modified_base_current, 'current_c')
        else:
            i1 = modified_base_current
            i2 = modified_base_current
            i3 = modified_base_current
            
        # Apply power factor noise
        base_pf = 0.95
        pf_load_impact = -0.05  # More load = worse power factor
        if has_noise:
            pf_noise = np.random.normal(0, 0.01)
            power_factor = min(0.98, max(0.85, base_pf + pf_load_impact + pf_noise))
        else:
            power_factor = min(0.98, max(0.85, base_pf + pf_load_impact))
            
        # Calculate powers for balanced three-phase system
        actual_voltage_avg = (v1 + v2 + v3) / 3
        actual_current_avg = (i1 + i2 + i3) / 3
        apparent_power = np.sqrt(3) * actual_voltage_avg * actual_current_avg  # S = √3 * V * I
        
        if has_noise:
            real_power = noise_generator.add_noise(apparent_power * power_factor, 'power_real')
            reactive_power = noise_generator.add_noise(apparent_power * np.sin(np.arccos(power_factor)), 'power_reactive')
        else:
            real_power = apparent_power * power_factor
            reactive_power = apparent_power * np.sin(np.arccos(power_factor))
        
        # Apply frequency noise
        if has_noise:
            frequency = noise_generator.add_noise(base_frequency, 'frequency')
        else:
            frequency = base_frequency
        
        measurements.update({
            'timestamp': timestamp,
            'V1': v1,
            'V2': v2,
            'V3': v3,
            'C1': i1,
            'C2': i2,
            'C3': i3,
            'V1_V2': v_ab,
            'V2_V3': v_bc,
            'V1_V3': v_ca,
            'frequency': frequency,
            'power_real': real_power,
            'power_effective': reactive_power,
            'power_apparent': apparent_power,
            'power_factor': power_factor
        })
        
        return measurements
        
    except Exception as e:
        print(f"Error in simulation: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

def main():
    try:
        args = parse_arguments()
        
        # Parse noise configurations
        noise_configs = []
        if not args.noise:
            # Default to gaussian noise if none specified
            noise_configs.append({
                'type': 'gaussian',
                'scale': 0.01
            })
        else:
            for noise_arg in args.noise:
                noise_configs.append(parse_noise_arg(noise_arg))
        
        # Add GMM configuration if reference file is provided
        if args.reference_file:
            gmm_config = {
                'type': 'gmm',
                'reference_file': args.reference_file,
                'n_components': args.gmm_components,
                'scale': 0.01
            }
            
            # Parse GMM parameters if no reference file
            if not args.reference_file:
                gmm_config.update({
                    'means': [float(x) for x in args.gmm_means.split(',')],
                    'variances': [float(x) for x in args.gmm_variances.split(',')],
                    'weights': [float(x) for x in args.gmm_weights.split(',')]
                })
            
            noise_configs.append(gmm_config)
        
        # Initialize noise generator with configurations
        noise_generator = NoiseGenerator(noise_configs)
        
        # Create noiseless noise generator
        no_noise_generator = NoiseGenerator([{'type': 'none'}])
        
        net = create_simple_network()
        
        start_time = datetime.now() 
        timestamps = [start_time + timedelta(seconds=i) for i in range(args.duration)]
        
        print(f"Generating {args.duration} seconds of data starting from {start_time}")
        print(f"Using voltage={args.voltage}V, current={args.current}A, frequency={args.frequency}Hz")
        
        # Display current pattern information
        print("Current pattern:")
        print("- First 20% of samples: base current")
        print("- Next 25% of samples: base current +20%")
        print("- Next 20% of samples: back to base current")
        print("- Next 20% of samples: base current -15%")
        print("- Final 15% of samples: back to base current")
        
        # Display noise configurations
        print("Noise configurations:")
        for i, config in enumerate(noise_configs):
            noise_type = config.get('type')
            scale = config.get('scale')
            print(f"  {i+1}. {noise_type} (scale={scale})")
            if noise_type == 'gmm' and config.get('reference_file'):
                print(f"     Using reference file: {config.get('reference_file')}")
        
        # Noiseless data
        print("Generating noiseless data...")
        noiseless_measurements = []
        for i, timestamp in enumerate(timestamps):
            measurements = simulate_measurements(
                net, 
                timestamp, 
                noise_generator=no_noise_generator,
                voltage=args.voltage,
                current=args.current,
                freq=args.frequency,
                current_index=i,
                total_duration=args.duration
            )
            noiseless_measurements.append(measurements)
        
        df_noiseless = pd.DataFrame(noiseless_measurements)
        df_noiseless.to_csv(args.output_noiseless, index=False)
        
        # Noisy data
        print("Generating noisy data...")
        noisy_measurements = []
        for i, timestamp in enumerate(timestamps):
            measurements = simulate_measurements(
                net, 
                timestamp, 
                noise_generator=noise_generator,
                voltage=args.voltage,
                current=args.current,
                freq=args.frequency,
                current_index=i,
                total_duration=args.duration
            )
            noisy_measurements.append(measurements)
        
        df_noisy = pd.DataFrame(noisy_measurements)
        df_noisy.to_csv(args.output_noisy, index=False)
        
        print("Data generation complete.")
        print("Generated files:")
        print(f"- {args.output_noisy} (with noise)")
        print(f"- {args.output_noiseless} (without noise)")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()