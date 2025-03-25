import pandas as pd

import matplotlib.pyplot as plt

# Load the data
df_autoencoder_mosaik = pd.read_csv('saves/autoencoder_data/Mosaik_autoencoderEPICnoise.csv')
df_autoencoder_panda = pd.read_csv('saves/autoencoder_data/Panda_autoencoderEPICnoise.csv')
df_raw_mosaik = pd.read_csv('raw_data/Mosaik.csv')
df_raw_panda = pd.read_csv('raw_data/Panda.csv')


column = "V1"
# Plot the data
plt.figure(figsize=(10, 6))

plt.plot(df_autoencoder_mosaik[column][5:], label='Autoencoder Mosaik')
plt.plot(df_autoencoder_panda[column][5:], label='Autoencoder Panda')
plt.plot(df_raw_mosaik[column][5:], label='Raw Mosaik')
plt.plot(df_raw_panda[column][5:], label='Raw Panda')

# Add labels and legend
plt.xlabel('Index')
plt.ylabel('V1')
plt.title('Comparison of Autoencoder Denoised and Raw Data')
plt.legend()

# Show the plot
plt.savefig('saves/autoencoder_vs_raw.png')