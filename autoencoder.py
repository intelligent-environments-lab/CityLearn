import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from sklearn.preprocessing import MinMaxScaler

# Source: https://github.com/techshot25/Autoencoders
class Autoencoder(nn.Module):
    """Makes the main denoising autoencoder

    Parameters
    ----------
    in_shape [int] : input shape
    enc_shape [int] : desired encoded shape
    """

    def __init__(self, in_shape, enc_shape):
        super(Autoencoder, self).__init__()

        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.encode = nn.Sequential(
            nn.Linear(in_shape, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(64, enc_shape),
        )
        
        self.decode = nn.Sequential(
            nn.BatchNorm1d(enc_shape),
            nn.Linear(enc_shape, 64),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, in_shape)
        )

        self.scaler = MinMaxScaler()
        self.error = nn.MSELoss()
        self.optimizer = Adam(self.parameters())
        
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def train_model(self, n_epochs, x, verbose = True):
        self.train()
        for epoch in range(1, n_epochs + 1):
            self.optimizer.zero_grad()
            output = self(x)
            loss = self.error(output, x)
            loss.backward()
            self.optimizer.step()
            if verbose:
                if epoch % int(0.1*n_epochs) == 0:
                    print(f'AE epoch {epoch} \t Loss: {loss.item():.4g}')
        print('\n')

    def encode_min(self, x):
        x = self.scaler.fit_transform([x])
        x = self.encode(torch.from_numpy(x).to(self.device))
        x = x.cpu().detach().numpy().tolist()[0]
        return x