import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Parameters
SIGNAL_LENGTH = 500  # Length of our ECG signals
BATCH_SIZE = 32
EPOCHS = 30
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
        return X_tensor, y_tensor

    def get_num_classes(self):
        return self.num_classes

# Variational Autoencoder Model
class VAE(nn.Module):
    def __init__(self, signal_length, latent_dim=2):  # Increased latent dimension
        super().__init__()
        self.encoder = nn.Sequential(
            # prvi konvolucijski blok
            nn.Conv1d(1, 16, kernel_size=5, stride=3, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # drugi konvolucijski blok
            nn.Conv1d(16, 16, kernel_size=5, stride=3, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # globalno povprečno združevanje
        )
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_logvar = nn.Linear(16, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, 16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(16, 16, kernel_size=signal_length//2, stride=signal_length//2),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=1),
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
def vae_loss(recon_x, x, mu, logvar, beta=0.01):
     recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
     kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
     return recon_loss + beta * kld, recon_loss, kld

# Training Setup
print("Starting training...")

# Load data (no split)
loader = LoadData('ecg-two-classes.npz', batch_size=BATCH_SIZE)
signal_length = loader.X.shape[-1]
data_loader = loader.get_loader(ToTensorDataset)
X_all, y_all = loader.get_all_data()

print(f"Signal length: {signal_length}")
print(f"Size of the dataset: {len(loader.X)}")
print()

model = VAE(signal_length, latent_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
# Training Loop with weighted KL divergence
beta = 0.005  # Weight for KL divergence
print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    total_recon = 0
    total_kld = 0
    for X_batch, _ in data_loader:
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
z_np = z.numpy()
y_all_np = y_all.numpy()
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

# Generate new signals
print("\nGenerating new signals...")
model.eval()
with torch.no_grad():
    # Sample from standard normal distribution in latent space
    num_samples = 3
    z_new = torch.randn(num_samples, 2)  # 2 is the latent dimension
    # Decode the samples
    new_signals = model.decode(z_new)
    
    # Plot the generated signals
    plt.figure(figsize=(15, 10))
    for i in range(num_samples):
        plt.subplot(num_samples, 1, i+1)
        signal = new_signals[i].squeeze().numpy()
        plt.plot(signal)
        plt.title(f'Generated Signal {i+1}')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig("generated-signals.svg")
    plt.show()
    print("Generated signals plot saved as 'generated-signals.svg'")

