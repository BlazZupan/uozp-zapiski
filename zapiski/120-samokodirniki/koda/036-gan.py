import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Parameters
SIGNAL_LENGTH = 500  # Length of our ECG signals
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0002
LATENT_DIM = 4

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

# Generator Model
class Generator(nn.Module):
    def __init__(self, latent_dim, signal_length):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, signal_length),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.model(z)
        x = x.unsqueeze(1)  # [batch, 1, signal_length]
        return x

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self, signal_length):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear((signal_length // 4) * 32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Training Setup
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data (no split)
loader = LoadData('ecg-two-classes.npz', batch_size=BATCH_SIZE)
signal_length = loader.X.shape[-1]
data_loader = loader.get_loader(ToTensorDataset)
X_all, y_all = loader.get_all_data(device)

# Filter to only those with y label of 0
mask = (y_all == 1)
X_all = X_all[mask]
y_all = y_all[mask]

# If you use data_loader, recreate it with the filtered data:
filtered_dataset = ToTensorDataset(X_all.cpu().numpy(), y_all.cpu().numpy())
data_loader = DataLoader(filtered_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Signal length: {signal_length}")
print(f"Size of the dataset: {len(loader.X)}")
print()

generator = Generator(LATENT_DIM, signal_length).to(device)
discriminator = Discriminator(signal_length).to(device)

criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

# Training Loop
print("Starting GAN training...")
for epoch in range(EPOCHS):
    generator.train()
    discriminator.train()
    total_g_loss = 0
    total_d_loss = 0
    for X_batch, _ in data_loader:
        X_batch = X_batch.to(device)
        batch_size = X_batch.size(0)

        # Real and fake labels
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        # Real
        outputs_real = discriminator(X_batch)
        d_loss_real = criterion(outputs_real, real_labels)
        # Fake
        z = torch.randn(batch_size, LATENT_DIM, device=device)
        fake_signals = generator(z)
        outputs_fake = discriminator(fake_signals.detach())
        d_loss_fake = criterion(outputs_fake, fake_labels)
        # Total loss
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()
        total_d_loss += d_loss.item()

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, LATENT_DIM, device=device)
        fake_signals = generator(z)
        outputs = discriminator(fake_signals)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizer_G.step()
        total_g_loss += g_loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, D Loss: {total_d_loss/len(data_loader):.4f}, G Loss: {total_g_loss/len(data_loader):.4f}")

# Generate and plot some fake signals
generator.eval()
with torch.no_grad():
    z = torch.randn(10, LATENT_DIM, device=device)
    fake_signals = generator(z).cpu().squeeze(1).numpy()

plt.figure(figsize=(12, 6))
# Plot 5 fake signals
for i in range(5):
    plt.plot(fake_signals[i] + i*2, color='red', label=f"Fake {i+1}" if i == 0 else "")
# Plot 5 real signals (from X_all)
real_signals = X_all.cpu().squeeze(1).numpy()
for i in range(5):
    plt.plot(real_signals[i] + (i+5)*2, color='black', label=f"Real {i+1}" if i == 0 else "")
plt.title("5 Fake (red) and 5 Real (black) ECG Signals (GAN)")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude (offset for clarity)")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.savefig("gan-generated-ecg-mixed.svg")
plt.show()
print("5 fake and 5 real ECG signals saved as 'gan-generated-ecg-mixed.svg'")

# Optionally, save the generator model
torch.save(generator.state_dict(), 'ecg_gan_generator.pth')
print("Generator model saved as 'ecg_gan_generator.pth'")
