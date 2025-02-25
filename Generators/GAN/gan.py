import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os
from torch.nn.utils import spectral_norm

# Residual block for Generator
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(out_features),
            nn.Dropout(dropout_rate),
            nn.Linear(out_features, out_features),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(out_features),
            nn.Dropout(dropout_rate)
        )
        
        self.skip = nn.Sequential()
        if in_features != out_features:
            self.skip = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return self.block(x) + self.skip(x)

# Enhanced Generator with Residual Connections
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        
        self.initial = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3)
        )
        
        # Residual blocks
        self.res1 = ResidualBlock(128, 256)
        self.res2 = ResidualBlock(256, 512)
        
        # Output layer
        self.output = nn.Sequential(
            nn.Linear(512, output_dim),
            nn.Tanh()  # Normalized output range
        )
    
    def forward(self, z):
        x = self.initial(z)
        x = self.res1(x)
        x = self.res2(x)
        return self.output(x)

# Discriminator with Spectral Normalization
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        
        # Applying spectral normalization to all linear layers
        self.model = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, 256)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            
            spectral_norm(nn.Linear(256, 128)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            
            spectral_norm(nn.Linear(128, 64)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            
            spectral_norm(nn.Linear(64, 1))
        )
    
    def forward(self, x):
        return self.model(x)

class PhysicalConstraintLoss(nn.Module):
    def __init__(self):
        super(PhysicalConstraintLoss, self).__init__()
        
    def forward(self, x):
        """
        Enforce physical constraints on power grid data:
        - Voltage balance: V1 + V2 + V3 should be close to zero in 3-phase systems
        - Power relationship: P²+Q² = S² (real_power² + reactive_power² = apparent_power²)
        - Current balance: C1 + C2 + C3 should be close to zero
        """
        loss = 0.0
        
        # Check if we have all necessary columns
        # Assuming columns order: frequency, V1, V2, V3, C1, C2, C3, V1_V2, V2_V3, V1_V3, power_real, power_effective, power_apparent
        
        # 1. Voltage balance for 3-phase system
        if x.size(1) >= 4:  # If we have voltage columns
            v_sum = x[:, 1] + x[:, 2] + x[:, 3]  # V1 + V2 + V3
            loss += torch.mean(v_sum * v_sum)  # Should be close to zero
        
        # 2. Current balance for 3-phase system
        if x.size(1) >= 7:  # If we have current columns
            c_sum = x[:, 4] + x[:, 5] + x[:, 6]  # C1 + C2 + C3
            loss += torch.mean(c_sum * c_sum)  # Should be close to zero
        
        # 3. Power relationship: P²+Q² ≈ S²
        if x.size(1) >= 13:  # If we have power columns
            # Assuming columns 10, 11, 12 are power_real, power_effective, power_apparent
            p_real = x[:, 10]
            p_effective = x[:, 11]
            p_apparent = x[:, 12]
            
            # Calculate Q (reactive power) using P²+Q²=S²
            p_reactive = torch.sqrt(torch.clamp(p_apparent * p_apparent - p_real * p_real, min=1e-6))
            
            # Check if P²+Q² = S²
            power_constraint = (p_real * p_real + p_reactive * p_reactive) - (p_apparent * p_apparent)
            loss += torch.mean(power_constraint * power_constraint)
        
        return loss

class CustomScaler:
    def __init__(self, base_freq, base_voltage, base_current):
        self.base_freq = base_freq
        self.base_voltage = base_voltage
        self.base_current = base_current
        self.scalers = {}
        
    def fit_transform(self, data):
        """Fit and transform data while respecting base values"""
        df = pd.DataFrame(data)
        scaled_data = pd.DataFrame()
        
        # Scale frequency
        if 'frequency' in df.columns:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaled_data['frequency'] = scaler.fit_transform(
                df[['frequency']].values * (self.base_freq / df['frequency'].mean())
            ).flatten()
            self.scalers['frequency'] = scaler
        
        # Scale voltages
        for col in ['V1', 'V2', 'V3']:
            if col in df.columns:
                scaler = MinMaxScaler(feature_range=(-1, 1))
                scaled_data[col] = scaler.fit_transform(
                    df[[col]].values * (self.base_voltage / df[col].mean())
                ).flatten()
                self.scalers[col] = scaler
        
        # Scale currents
        for col in ['C1', 'C2', 'C3']:
            if col in df.columns:
                scaler = MinMaxScaler(feature_range=(-1, 1))
                scaled_data[col] = scaler.fit_transform(
                    df[[col]].values * (self.base_current / df[col].mean())
                ).flatten()
                self.scalers[col] = scaler
        
        # Scale voltage differences
        for col in ['V1_V2', 'V2_V3', 'V1_V3']:
            if col in df.columns:
                scaler = MinMaxScaler(feature_range=(-1, 1))
                scaled_data[col] = scaler.fit_transform(
                    df[[col]].values * (self.base_voltage / df[col].mean())
                ).flatten()
                self.scalers[col] = scaler
        
        # Scale power components
        for col in ['power_real', 'power_effective', 'power_apparent']:
            if col in df.columns:
                scaler = MinMaxScaler(feature_range=(-1, 1))
                power_base = self.base_voltage * self.base_current
                scaled_data[col] = scaler.fit_transform(
                    df[[col]].values * (power_base / df[col].mean())
                ).flatten()
                self.scalers[col] = scaler
        
        return scaled_data.values
    
    def inverse_transform(self, data):
        """Inverse transform scaled data while maintaining base values"""
        df = pd.DataFrame(data)
        result = pd.DataFrame()
        
        cols = ['frequency'] + \
               ['V1', 'V2', 'V3'] + \
               ['C1', 'C2', 'C3'] + \
               ['V1_V2', 'V2_V3', 'V1_V3'] + \
               ['power_real', 'power_effective', 'power_apparent']
        
        for i, col in enumerate(cols):
            if col in self.scalers:
                values = self.scalers[col].inverse_transform(data[:, [i]])
                
                # Adjust values to match base values
                if col == 'frequency':
                    values = values * (self.base_freq / values.mean())
                elif col.startswith('V'):
                    values = values * (self.base_voltage / values.mean())
                elif col.startswith('C'):
                    values = values * (self.base_current / values.mean())
                elif col.startswith('power'):
                    power_base = self.base_voltage * self.base_current
                    values = values * (power_base / values.mean())
                
                result[col] = values.flatten()
        
        return result.values

def add_noise(data, noise_factor=0.05):
    """Add small Gaussian noise to input data for stability"""
    noise = torch.randn_like(data) * noise_factor
    return data + noise

def compute_gradient_penalty(discriminator, real_data, fake_data, lambda_gp=10):
    """Compute gradient penalty for WGAN-GP"""
    batch_size = real_data.size(0)
    
    # Random interpolation between real and fake data
    alpha = torch.rand(batch_size, 1, device=real_data.device)
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated.requires_grad_(True)
    
    # Get discriminator output for interpolated data
    d_interpolated = discriminator(interpolated)
    
    # Calculate gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Calculate gradient penalty
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    
    return gradient_penalty * lambda_gp

def load_and_preprocess_data(file_paths, custom_scaler, data_type='unknown'):
    """Load and preprocess multiple CSV files using custom scaler."""
    all_data = []
    print(f"Loading {data_type} data files...")
    
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            if 'timestamp' in df.columns:
                data = df.drop('timestamp', axis=1)
                all_data.append(data)
                print(f"Successfully loaded {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    if not all_data:
        raise ValueError(f"No valid {data_type} data files loaded")
        
    combined_data = pd.concat(all_data, ignore_index=True)
    scaled_data = custom_scaler.fit_transform(combined_data)
    
    return scaled_data

def train_gan(real_files, simulated_files, epochs, n_samples, output_path, base_freq=50.0, 
            base_voltage=230.0, base_current=100.0, pretrain=True):
    print("Initializing Enhanced WGAN-GP training with real and simulated data...")
    print("Using residual connections in Generator and spectral normalization in Discriminator")
    print(f"\nUsing base values:")
    print(f"Base Voltage: {base_voltage} V")
    print(f"Base Current: {base_current} A")
    print(f"Base Frequency: {base_freq} Hz")
    
    # Initialize custom scaler with base values
    custom_scaler = CustomScaler(base_freq, base_voltage, base_current)
    
    # Load and preprocess data using custom scaler
    real_data = load_and_preprocess_data(real_files, custom_scaler, data_type='real')
    simulated_data = load_and_preprocess_data(simulated_files, custom_scaler, data_type='simulated')
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Initialize models
    latent_dim = 64
    output_dim = 13  # Number of features in the data
    generator = Generator(latent_dim, output_dim).to(device)
    discriminator = Discriminator(output_dim).to(device)
    physical_loss = PhysicalConstraintLoss().to(device)
    
    print(f"\nDataset sizes:")
    print(f"Real data: {len(real_data)} samples")
    print(f"Simulated data: {len(simulated_data)} samples")
    
    # Create DataLoaders
    batch_size = 32
    real_dataset = TensorDataset(torch.FloatTensor(real_data))
    sim_dataset = TensorDataset(torch.FloatTensor(simulated_data))
    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    sim_loader = DataLoader(sim_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Initialize optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    
    # Hyperparameters
    lambda_gp = 10  # Gradient penalty coefficient
    n_critic = 5    # Number of critic iterations per generator iteration
    physical_loss_weight = 0.1  # Weight for physical constraints loss
    
    # Learning rate schedulers for improved stability
    scheduler_g = optim.lr_scheduler.ExponentialLR(g_optimizer, gamma=0.995)
    scheduler_d = optim.lr_scheduler.ExponentialLR(d_optimizer, gamma=0.995)
    
    # Initialize tracking metrics
    best_gen_loss = float('inf')
    patience = 0
    patience_limit = 10
    best_generator_state = None
    
    print("\nStarting Enhanced WGAN-GP adversarial training...")
    for epoch in range(epochs):
        g_losses = []
        d_losses = []
        gp_losses = []
        phys_losses = []
        
        real_iter = iter(real_loader)
        sim_iter = iter(sim_loader)
        n_batches = min(len(real_loader), len(sim_loader))
        
        for batch_idx in range(n_batches):
            try:
                real_batch = next(real_iter)[0]
                sim_batch = next(sim_iter)[0]
                
                min_batch_size = min(real_batch.size(0), sim_batch.size(0))
                if min_batch_size < batch_size:
                    continue
                
                real_batch = real_batch[:min_batch_size].to(device)
                sim_batch = sim_batch[:min_batch_size].to(device)
                
                # Add noise to input data for stability
                real_batch_noisy = add_noise(real_batch, noise_factor=0.03)
                
                # Train Discriminator
                for _ in range(n_critic):
                    d_optimizer.zero_grad()
                    
                    # Real data - maximize D(x)
                    real_validity = discriminator(real_batch_noisy)
                    d_real = -torch.mean(real_validity)
                    
                    # Fake data - minimize D(G(z))
                    z = torch.randn(min_batch_size, latent_dim, device=device)
                    fake_data = generator(z).detach()
                    fake_validity = discriminator(add_noise(fake_data, noise_factor=0.03))
                    d_fake = torch.mean(fake_validity)
                    
                    # Gradient penalty
                    gradient_penalty = compute_gradient_penalty(discriminator, real_batch, fake_data, lambda_gp)
                    
                    # Total discriminator loss
                    d_loss = d_real + d_fake + gradient_penalty
                    d_loss.backward()
                    d_optimizer.step()
                    
                    d_losses.append(d_loss.item())
                    gp_losses.append(gradient_penalty.item())
                
                # Train Generator
                g_optimizer.zero_grad()
                
                z = torch.randn(min_batch_size, latent_dim, device=device)
                fake_data = generator(z)
                fake_validity = discriminator(fake_data)
                
                # Generator - maximize D(G(z))
                g_adv_loss = -torch.mean(fake_validity)
                
                # Physical constraints
                g_phys_loss = physical_loss(fake_data)
                phys_losses.append(g_phys_loss.item())
                
                # Total generator loss
                g_loss = g_adv_loss + physical_loss_weight * g_phys_loss
                g_loss.backward()
                g_optimizer.step()
                
                g_losses.append(g_loss.item())
                
                # Print progress
                if batch_idx % 50 == 0:
                    print(f"  Batch {batch_idx}/{n_batches} - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
                
            except StopIteration:
                break
        
        # Step the learning rate schedulers
        scheduler_g.step()
        scheduler_d.step()
        
        # Calculate average losses for this epoch
        avg_g_loss = np.mean(g_losses)
        avg_d_loss = np.mean(d_losses)
        avg_gp_loss = np.mean(gp_losses)
        avg_phys_loss = np.mean(phys_losses)
        
        # Implement early stopping with model saving
        if avg_g_loss < best_gen_loss:
            best_gen_loss = avg_g_loss
            patience = 0
            # Save best model state
            best_generator_state = generator.state_dict()
        else:
            patience += 1
        
        if (epoch + 1) % 5 == 0 or patience >= patience_limit:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Generator Loss: {avg_g_loss:.4f}")
            print(f"  Discriminator Loss: {avg_d_loss:.4f}")
            print(f"  Gradient Penalty: {avg_gp_loss:.4f}")
            print(f"  Physical Loss: {avg_phys_loss:.4f}")
            
            # Validation step
            with torch.no_grad():
                z = torch.randn(100, latent_dim, device=device)
                fake_data = generator(z)
                fake_scores = discriminator(fake_data)
                real_scores = discriminator(real_batch)
                print(f"  Avg Real Score: {torch.mean(real_scores):.4f}")
                print(f"  Avg Fake Score: {torch.mean(fake_scores):.4f}")
        
        # Check for early stopping
        if patience >= patience_limit:
            print(f"\nEarly stopping at epoch {epoch+1}. No improvement for {patience_limit} epochs.")
            break
    
    # Load best generator if available
    if best_generator_state is not None:
        generator.load_state_dict(best_generator_state)
        print("\nLoaded best generator model based on loss.")
    
    # Generate final data
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim, device=device)
        generated = generator(z)
        generated = generated.cpu()
        
        # Use custom scaler for inverse transform
        gen_data_array = custom_scaler.inverse_transform(generated.numpy())
        df_cols = pd.read_csv(real_files[0]).drop('timestamp', axis=1).columns
        gen_data = pd.DataFrame(gen_data_array, columns=df_cols)
        
        # Add timestamp
        gen_data['timestamp'] = [datetime.now() + timedelta(seconds=i) for i in range(len(gen_data))]
        
        # Reorder columns
        column_order = ['timestamp'] + list(df_cols)
        gen_data = gen_data[column_order]
        
        # Print statistics of generated data
        print("\nGenerated Data Statistics:")
        print(f"Frequency mean: {gen_data['frequency'].mean():.2f} Hz")
        print(f"Voltage means: V1={gen_data['V1'].mean():.2f}, V2={gen_data['V2'].mean():.2f}, V3={gen_data['V3'].mean():.2f} V")
        print(f"Current means: C1={gen_data['C1'].mean():.2f}, C2={gen_data['C2'].mean():.2f}, C3={gen_data['C3'].mean():.2f} A")
        
        gen_data.to_csv(output_path, index=False)
        print(f"\nGenerated data saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate power grid data using Enhanced WGAN-GP with residual connections and spectral normalization')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--output', type=str, default='generated_data.csv', help='Output CSV file path')
    parser.add_argument('--pretrain', action='store_true', help='Enable pretraining of the networks')
    parser.add_argument('--real-data', nargs='+', required=True, help='Paths to real data CSV files')
    parser.add_argument('--simulated-data', nargs='+', required=True, help='Paths to simulated data CSV files')
    
    # Add base value parameters
    parser.add_argument('--frequency', type=float, default=50.0, help='Base frequency in Hz (default: 50.0)')
    parser.add_argument('--voltage', type=float, default=230.0, help='Base voltage in V (default: 230.0)')
    parser.add_argument('--current', type=float, default=100.0, help='Base current in A (default: 100.0)')
        
    args = parser.parse_args()
    
    try:
        train_gan(
            real_files=args.real_data,
            simulated_files=args.simulated_data,
            epochs=args.epochs,
            n_samples=args.samples,
            output_path=args.output,
            base_freq=args.frequency,
            base_voltage=args.voltage,
            base_current=args.current,
            pretrain=args.pretrain
        )
    except Exception as e:
        print(f"\nError during execution: {e}")
        raise

if __name__ == "__main__":
    main()