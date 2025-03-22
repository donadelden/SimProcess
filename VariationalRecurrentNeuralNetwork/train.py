import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import matplotlib.pyplot as plt 
from model import VRNN
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

"""implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""

def train(epoch):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):

        #transforming data
        data = data.to(device)
        data = data.squeeze().transpose(0, 1) # (seq, batch, elem)
        data = (data - data.min()) / (data.max() - data.min())
        
        #forward + backward + optimize
        optimizer.zero_grad()
        kld_loss, nll_loss, _, _ = model(data)
        loss = kld_loss + nll_loss
        loss.backward()
        optimizer.step()

        #grad norm clipping, only in pytorch version >= 1.10
        nn.utils.clip_grad_norm_(model.parameters(), clip)

        #printing
        if batch_idx % print_every == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t KLD Loss: {:.6f} \t NLL Loss: {:.6f}'.format(
                epoch, batch_idx * batch_size, batch_size * (len(train_loader.dataset)//batch_size),
                100. * batch_idx / len(train_loader),
                kld_loss / batch_size,
                nll_loss / batch_size))
            
            # sample = model.sample(torch.tensor(28, device=device))
            # plt.imshow(sample.to(torch.device('cpu')).numpy())
            # plt.pause(1e-6)

        train_loss += loss.item()

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))
    
    return train_loss / len(train_loader.dataset)
    

def test(epoch):
    """uses test data to evaluate 
    likelihood of the model"""

    mean_kld_loss, mean_nll_loss = 0, 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):                                            

            data = data.to(device)
            data = data.squeeze().transpose(0, 1)
            data = (data - data.min()) / (data.max() - data.min())

            kld_loss, nll_loss, _, _ = model(data)
            mean_kld_loss += kld_loss.item()
            mean_nll_loss += nll_loss.item()

    # Calculate the average Kullback-Leibler divergence (KLD) loss
    mean_kld_loss /= len(test_loader.dataset)
    
    # Calculate the average negative log-likelihood (NLL) loss
    mean_nll_loss /= len(test_loader.dataset)
   
    print('====> Test set loss: KLD Loss = {:.4f}, NLL Loss = {:.4f} '.format(
        mean_kld_loss, mean_nll_loss))
    
    return mean_kld_loss, mean_nll_loss

def create_sequences(data, n):
    X, y = [], []
    for i in range(len(data) - n):
        X.append(data[i:i+n])  # Previous (n-1) samples)
        y.append(data[i+n])    # Predict the nth sample
    return np.array(X), np.array(y)

def load_data(file_paths, window_size):
    X, y = None, None
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        train_data = df[measurements_to_keep]
        train_data_np = np.array(train_data)

        X_tmp, y_tmp = create_sequences(train_data_np, window_size)
        if X is None:
            X = X_tmp
            y = y_tmp
        else:
            X = np.concatenate((X, X_tmp), axis=0)
            y = np.concatenate((y, y_tmp), axis=0)

    # print("X shape:", X.shape)  # (samples, time_steps, measurements)
    # print(X[0])
    # print("y shape:", y.shape)  # (samples,)
    # print(y[0])

    return X, y



if __name__ == '__main__':

    window_size = 10
    #measurements_to_keep = ['frequency', 'V1', 'V2', 'V3', 'C1', 'C2', 'C3', 'V1_V2', 'V2_V3', 'V1_V3', 'power_real', 'power_effective', 'power_apparent']  # ['frequency','V1','V2','V3','C1','C2','C3']  # 7
    measurements_to_keep = ["C1_noise", "C2_noise", "C3_noise", "V1_noise", "V2_noise", "V3_noise", "frequency_noise", "power_real_noise", "power_effective_noise", "power_apparent_noise"]
    measurements_number = len(measurements_to_keep)
    x_in_dim = measurements_number
    file_paths = [ f'raw_data/EPIC{i}_noise.csv' for i in range(1, 9)]  # <--- DEFINE SOURCE FILE PATHS HERE

    #hyperparameters
    x_dim = measurements_number #28   # input dimensions
    h_dim = 100     # hidden layer dimensions
    z_dim = 16      # latent space dimensions
    n_layers =  5   # number of layers
    n_epochs = 41
    clip = 10
    learning_rate = 1e-3
    batch_size = 8 #128
    seed = 42
    print_every = 1000 # batches
    save_every = 10 # epochs


    # changing device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')

    #manual seed
    torch.manual_seed(seed)
    plt.ion()

    #init model + optimizer + datasets

    X, _ = load_data(file_paths, window_size)

    # Split the data into train and test sets
    X_train, X_test = train_test_split(X, test_size=0.1, random_state=seed)

    print(X_train.shape, X_test.shape)

    # Create DataLoader for train and test sets
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                                torch.ones(X_train.shape[0], dtype=torch.float32))
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32), 
                                                torch.ones(X_test.shape[0], dtype=torch.float32))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = VRNN(x_dim, h_dim, z_dim, n_layers)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Prepare a dataframe to save losses
    losses_df = pd.DataFrame(columns=['epoch', 'train_loss', 'kld_loss', 'nll_loss'])

    for epoch in range(1, n_epochs + 1):

        #training + testing
        train_loss = train(epoch)
        kld_loss, nll_loss = test(epoch)

        # Append losses to the dataframe
        new_row = pd.DataFrame({'epoch': [epoch], 'train_loss': [train_loss], 'kld_loss': [kld_loss], 'nll_loss': [nll_loss]})
        losses_df = pd.concat([losses_df, new_row], ignore_index=True)

        #saving model
        if epoch % save_every == 1:
            fn = 'saves/vrnn_state_dict_'+str(epoch)+'.pth'
            torch.save(model.state_dict(), fn)
            print('Saved model to '+fn)

    # Save the losses dataframe to a CSV file
    losses_df.to_csv('saves/losses_all.csv', index=False)