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
        description='Generate power network measurements data'
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

def simulate_measurements(net, timestamp, add_noise=False, freq=50.0, voltage=245.0, current=20.0):
    """Simulate electrical measurements with optional noise"""
    try:
        pp.runpp(net, calculate_voltage_angles=True)
        
        measurements = {}
        base_frequency = freq
        
        bus = 0
        vm_pu = net.res_bus.vm_pu[bus]
        t = timestamp.timestamp()
        
        daily_var = np.sin(2 * np.pi * t / (24 * 3600)) * 0.1
        hourly_var = np.sin(2 * np.pi * t / 3600) * 0.05
        fast_var = np.sin(2 * np.pi * t / 300) * 0.02
        
        load_variation = 1.0 + daily_var + hourly_var + fast_var
        
        base_voltage = voltage
        voltage_variation = vm_pu * (1 + np.sin(2 * np.pi * t / 600) * 0.01)
        
        angle_1 = 0 + (normal(0, 0.5) if add_noise else 0)
        angle_2 = -120 + (normal(0, 0.5) if add_noise else 0)
        angle_3 = 120 + (normal(0, 0.5) if add_noise else 0)
        
        v1 = base_voltage * voltage_variation * (1 + (normal(0, 0.005) if add_noise else 0))
        v2 = base_voltage * voltage_variation * (1 + (normal(0, 0.005) if add_noise else 0))
        v3 = base_voltage * voltage_variation * (1 + (normal(0, 0.005) if add_noise else 0))
        
        base_current = current * load_variation
        current_noise = 0.05 if add_noise else 0
        i1 = base_current * (1 + normal(0, current_noise) if add_noise else 1)
        i2 = base_current * (1 + normal(0, current_noise) if add_noise else 1)
        i3 = base_current * (1 + normal(0, current_noise) if add_noise else 1)
        
        # Power factor calculation
        base_pf = 0.95
        pf_load_impact = -0.05 * (load_variation - 1)
        power_factor = min(0.98, max(0.85, base_pf + pf_load_impact))
        
        apparent_power = np.sqrt(3) * sum([v * i for v, i in zip([v1, v2, v3], [i1, i2, i3])]) / 3
        real_power = apparent_power * power_factor
        reactive_power = apparent_power * np.sqrt(1 - power_factor**2)


        
        freq_base = np.sin(2 * np.pi * t / 3600) * 0.02
        freq_load = -0.01 * (load_variation - 1)
        freq_noise = normal(0, 0.005) if add_noise else 0
        frequency = base_frequency + freq_base + freq_load + freq_noise
        
        measurements.update({
            'timestamp': timestamp,
            'V1': v1,
            'V2': v2,
            'V3': v3,
            'C1': i1,
            'C2': i2,
            'C3': i3,
            'V1_V2': v1 - v2,
            'V2_V3': v2 - v3,
            'V1_V3': v1 - v3,
            'frequency': frequency,
            'power_real': real_power,
            'power_effective': reactive_power,
            'power_apparent': apparent_power
        })
        
        return measurements
        
    except Exception as e:
        print(f"Error in simulation: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

def main():
    try:
        args = parse_arguments()
        net = create_simple_network()
        
        start_time = datetime.now() 
        timestamps = [start_time + timedelta(seconds=i) for i in range(args.duration)]
        
        print(f"Generating {args.duration} seconds of data starting from {start_time}")
        print(f"Using voltage={args.voltage}V, current={args.current}A, frequency={args.frequency}Hz")
        
        # Noiseless data
        noiseless_measurements = []
        for timestamp in timestamps:
            measurements = simulate_measurements(
                net, 
                timestamp, 
                add_noise=False,
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
                add_noise=True,
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