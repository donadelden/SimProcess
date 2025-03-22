import torch
from model import VRNN
import pandas as pd

if __name__ == "__main__":
    #hyperparameters
    window_size = 10
    measurements_to_keep = ["C1", "C2", "C3", "V1", "V2", "V3", "frequency", "power_real", "power_effective", "power_apparent"]
    measurements_number = len(measurements_to_keep)
    x_in_dim = measurements_number

    x_dim = measurements_number #28   # input dimensions
    h_dim = 100
    z_dim = 16
    n_layers =  5

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(f"Measurements count: {measurements_number}")

    state_dict = torch.load('saves/vrnn_state_dict_41.pth')
    model = VRNN(x_dim, h_dim, z_dim, n_layers)
    model.load_state_dict(state_dict)
    model.to(device)


    # Load simulated data from CSV
    filenames = ["Mosaik.csv", "Panda.csv"]

    for filename in filenames:
        data_path = f'raw_data/{filename}'
        data = pd.read_csv(data_path)

        # Filter the data to keep only the required measurements
        filtered_data = data[measurements_to_keep]

        # Convert the filtered data to a numpy array
        inputs = filtered_data.values.tolist()

        # Increase input_weith to 0.99 to get samples that are more similar to the input
        samples = model.sample(len(inputs), torch.tensor(inputs, device=device), input_weight=0.99)

        print(f"Input[1]: {inputs[1]}")
        print(f"Sample[1]: {samples[1]}")

        # Convert samples to numpy array and save to CSV
        samples_np = samples.cpu().detach().numpy()
        samples_np = samples_np[1:]
        samples_df = pd.DataFrame(samples_np, columns=measurements_to_keep)
        samples_df.to_csv(f'saves/output/{filename[:-4]}_autoencoderEPICnoise.csv', index=False)