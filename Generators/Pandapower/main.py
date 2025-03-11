import pandapower as pp
import pandas as pd
import numpy as np
from numpy.random import normal
from datetime import datetime, timedelta
import sys
import traceback
import argparse

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
    # Add noise type selection
    parser.add_argument(
        '--noise-type', 
        type=str, 
        default='gaussian',
        choices=['gaussian', 'uniform', 'laplace', 'poisson', 'impulse', 'brownian', 'pink', 'none'],
        help='Type of noise to apply (default: gaussian)'
    )
    # Add noise scale parameter
    parser.add_argument(
        '--noise-scale', 
        type=float, 
        default=0.01,
        help='Scale/intensity of the noise (default: 0.01)'
    )
    # Add impulse probability
    parser.add_argument(
        '--impulse-prob', 
        type=float, 
        default=0.01,
        help='Probability of impulse noise occurrence (default: 0.01)'
    )
    # Add poisson lambda
    parser.add_argument(
        '--poisson-lambda', 
        type=float, 
        default=2.0,
        help='Lambda parameter for Poisson noise (default: 2.0)'
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

def add_noise(base_value, noise_config, **kwargs):
    """
    Add noise to a base value using specified noise model.
    
    Parameters:
    -----------
    base_value : float
        Original value to add noise to
    noise_config : dict
        Configuration dictionary for the noise
    **kwargs : dict
        Additional parameters for specific noise types
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
        
    elif noise_type == 'none':
        return base_value
        
    else:
        # Default to no noise for unknown types
        return base_value

def simulate_measurements(net, timestamp, noise_config, freq=50.0, voltage=245.0, current=20.0):
    """Simulate electrical measurements with configurable noise"""
    try:
        pp.runpp(net, calculate_voltage_angles=True)
        
        measurements = {}
        base_frequency = freq
        
        bus = 0
        t = timestamp.timestamp()
        
        # Get base voltage variation (applies to all phases)
        voltage_variation = 1.0
        if noise_config['type'] != 'none':
            voltage_variation = add_noise(1.0, noise_config)
        
        # Apply phase angle variations
        if noise_config['type'] != 'none':
            angle_1 = 0 + np.random.normal(0, 0.1)
            angle_2 = -120 + np.random.normal(0, 0.1)
            angle_3 = 120 + np.random.normal(0, 0.1)
        else:
            angle_1 = 0
            angle_2 = -120
            angle_3 = 120
            
        # Apply voltage noise to each phase
        base_voltage = voltage
        if noise_config['type'] != 'none':
            v1 = add_noise(base_voltage * voltage_variation, noise_config)
            v2 = add_noise(base_voltage * voltage_variation, noise_config)
            v3 = add_noise(base_voltage * voltage_variation, noise_config)
        else:
            v1 = base_voltage * voltage_variation
            v2 = base_voltage * voltage_variation
            v3 = base_voltage * voltage_variation
            
        # Calculate line-to-line voltages (direct differences as in the original code)
        v_ab = v1 - v2
        v_bc = v2 - v3
        v_ca = v1 - v3
        
        # Apply current noise to each phase
        base_current = current
        if noise_config['type'] != 'none':
            i1 = add_noise(base_current, noise_config, scale=0.05)
            i2 = add_noise(base_current, noise_config, scale=0.05)
            i3 = add_noise(base_current, noise_config, scale=0.05)
        else:
            i1 = base_current
            i2 = base_current
            i3 = base_current
            
        # Apply power factor noise
        base_pf = 0.95
        pf_load_impact = -0.05  # More load = worse power factor
        if noise_config['type'] != 'none':
            pf_noise = np.random.normal(0, 0.01)
            power_factor = min(0.98, max(0.85, base_pf + pf_load_impact + pf_noise))
        else:
            power_factor = min(0.98, max(0.85, base_pf + pf_load_impact))
            
        # Calculate powers for balanced three-phase system
        actual_voltage_avg = (v1 + v2 + v3) / 3
        actual_current_avg = (i1 + i2 + i3) / 3
        apparent_power = np.sqrt(3) * actual_voltage_avg * actual_current_avg  # S = √3 * V * I
        real_power = apparent_power * power_factor  # P = S * cos(φ)
        reactive_power = apparent_power * np.sin(np.arccos(power_factor))  # Q = S * sin(φ)
        
        # Apply frequency noise
        if noise_config['type'] != 'none':
            frequency = add_noise(base_frequency, noise_config, scale=0.005)
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
        
        # Configure noise parameters based on command line arguments
        noise_config = {
            'type': args.noise_type,
            'scale': args.noise_scale,
            'impulse_prob': args.impulse_prob,
            'poisson_lambda': args.poisson_lambda
        }
        
        net = create_simple_network()
        
        start_time = datetime.now() 
        timestamps = [start_time + timedelta(seconds=i) for i in range(args.duration)]
        
        print(f"Generating {args.duration} seconds of data starting from {start_time}")
        print(f"Using voltage={args.voltage}V, current={args.current}A, frequency={args.frequency}Hz")
        print(f"Noise model: {args.noise_type} with scale {args.noise_scale}")
        
        # Create noiseless configuration
        no_noise_config = {'type': 'none', 'scale': 0}
        
        # Noiseless data
        noiseless_measurements = []
        for timestamp in timestamps:
            measurements = simulate_measurements(
                net, 
                timestamp, 
                noise_config=no_noise_config,
                voltage=args.voltage,
                current=args.current,
                freq=args.frequency
            )
            noiseless_measurements.append(measurements)
        
        df_noiseless = pd.DataFrame(noiseless_measurements)
        df_noiseless.to_csv('Panda_denoised.csv', index=False)
        
        # Noisy data
        noisy_measurements = []
        for timestamp in timestamps:
            measurements = simulate_measurements(
                net, 
                timestamp, 
                noise_config=noise_config,
                voltage=args.voltage,
                current=args.current,
                freq=args.frequency
            )
            noisy_measurements.append(measurements)
        
        df_noisy = pd.DataFrame(noisy_measurements)
        df_noisy.to_csv('Panda.csv', index=False)
        
        print("Data generation complete.")
        print("Generated files:")
        print("- Panda.csv (with noise)")
        print("- Panda_denoised.csv (without noise)")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()