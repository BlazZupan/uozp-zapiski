import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Parameters
SIGNAL_LENGTH = 500  # Length of our ECG signals
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.01

# Custom Dataset
class ToTensorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        if self.X.ndim == 2:
            self.X = self.X.unsqueeze(1)  # Add channel dimension if needed
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class LoadData:
    def __init__(self, filename, batch_size=32):
        data = np.load(filename)
        X = data['X']
        y = data['y']
        self.batch_size = batch_size
        self.X = X
        self.y = y
        self.num_classes = len(np.unique(y))

    def get_loader(self, ToTensorDataset):
        dataset = ToTensorDataset(self.X, self.y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return loader

    def get_all_data(self, device=None):
        import torch
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        if X_tensor.ndim == 2:
            X_tensor = X_tensor.unsqueeze(1)
        y_tensor = torch.tensor(self.y, dtype=torch.long)
        if device is not None:
            X_tensor = X_tensor.to(device)
            y_tensor = y_tensor.to(device)
        return X_tensor, y_tensor

    def get_num_classes(self):
        return self.num_classes

# Variational Autoencoder Model
class VAE(nn.Module):
    def __init__(self, signal_length, latent_dim=10):  # Increased latent dimension
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
            nn.Dropout(0.5)  # Added dropout
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, 32)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 32, kernel_size=signal_length//2, stride=signal_length//2),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=1),
        )
        self.signal_length = signal_length

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_fc(z)
        x = x.unsqueeze(-1)  # [batch, 32, 1]
        x = x.repeat(1, 1, self.signal_length // 2)
        x = self.decoder(x)
        x = x[:, :, :self.signal_length]  # Ensure output length matches input
        return x
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z


# VAE Loss function
def vae_loss(recon_x, x, mu, logvar, beta=10.0):
     recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
     kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
     return recon_loss + beta * kld, recon_loss, kld

# def vae_loss(recon_x, x, mu, logvar):
#     # Reconstruction loss (MSE)
#     recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
#     # KL divergence
#     kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
#     return recon_loss + kld, recon_loss, kld

# Training Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data (no split)
loader = LoadData('ecg-two-classes.npz', batch_size=BATCH_SIZE)
signal_length = loader.X.shape[-1]
data_loader = loader.get_loader(ToTensorDataset)
X_all, y_all = loader.get_all_data(device)

print(f"Signal length: {signal_length}")
print(f"Size of the dataset: {len(loader.X)}")
print()

model = VAE(signal_length, latent_dim=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
# Training Loop with weighted KL divergence
beta = 5.0  # Weight for KL divergence
print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    total_recon = 0
    total_kld = 0
    for X_batch, _ in data_loader:
        X_batch = X_batch.to(device)
        optimizer.zero_grad()
        X_recon, mu, logvar, _ = model(X_batch)
        loss, recon_loss, kld = vae_loss(X_recon, X_batch, mu, logvar, beta=beta)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kld += kld.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(data_loader):.4f}, "
          f"Recon: {total_recon/len(data_loader):.4f}, KLD: {total_kld/len(data_loader):.4f}")

# Evaluation: Reconstruction error on all data
print("\nEvaluating reconstruction on all data...")
model.eval()
with torch.no_grad():
    X_recon, mu, logvar, z = model(X_all)
    loss, recon_loss, kld = vae_loss(X_recon, X_all, mu, logvar)
print(f"Total VAE Loss: {loss:.6f} | Recon Loss: {recon_loss:.6f} | KLD: {kld:.6f}")

# Plot 2D bottleneck embeddings colored by class
z_np = z.cpu().numpy()
y_all_np = y_all.cpu().numpy()
plt.figure(figsize=(8, 6))
for c in np.unique(y_all_np):
    plt.scatter(z_np[y_all_np == c, 0], z_np[y_all_np == c, 1], label=f"Class {c}", alpha=0.6)
plt.xlabel("Latent dim 1")
plt.ylabel("Latent dim 2")
plt.title("2D Latent Embeddings of ECG VAE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("vae-autoencoder-two.svg")
plt.show()
print("2D embedding plot saved as 'vae-autoencoder-two.svg'")

# Pick a random input signal and show the reconstruction
# idx = 30
# X_random = X_all[idx:idx+1].to(device)  # shape [1, 1, signal_length]
# with torch.no_grad():
#     X_recon, _, _, _ = model(X_random)
# X_orig = X_random.cpu().squeeze().numpy()
# X_recon = X_recon.cpu().squeeze().numpy()

# plt.figure(figsize=(10, 4))
# plt.plot(X_orig, label='Original')
# plt.plot(X_recon, label='Reconstructed')
# plt.legend()
# plt.title(f"ECG Signal Reconstruction (Sample {idx})")
# plt.xlabel("Sample Index")
# plt.ylabel("Amplitude")
# plt.savefig("reconstruction-vae-two.svg")
# plt.show()
