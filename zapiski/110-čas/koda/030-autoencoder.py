import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# parametri
SIGNAL_LENGTH = 500  # dolžina naših ekg signalov
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.005

# prilagojen nabor podatkov
class ToTensorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        if self.X.ndim == 2:
            self.X = self.X.unsqueeze(1)  # dodaj dimenzijo kanala, če je potrebno
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

    def get_all_data(self):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        if X_tensor.ndim == 2:
            X_tensor = X_tensor.unsqueeze(1)
        y_tensor = torch.tensor(self.y, dtype=torch.long)
        return X_tensor, y_tensor

    def get_num_classes(self):
        return self.num_classes

# model avtokodirnika
class Autoencoder(nn.Module):
    def __init__(self, signal_length):
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
        self.bottleneck = nn.Linear(16, 2)
        self.decoder_fc = nn.Linear(2, 16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(16, 16, kernel_size=signal_length//2, stride=signal_length//2),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=1),
        )
        self.signal_length = signal_length

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1) # iz [batch, 16, 1] v [batch, 16]
        z = self.bottleneck(x)
        x = self.decoder_fc(z)
        x = x.unsqueeze(-1)  # [batch, 16, 1]
        # povečaj na [batch, 16, signal_length//2]
        x = x.repeat(1, 1, self.signal_length // 2)
        x = self.decoder(x)
        x = x[:, :, :self.signal_length]  # zagotovi, da dolžina izhoda ustreza vhodu
        return x, z

# nalaganje podatkov (brez razdelitve)
# loader = LoadData('ecg-four-classes.npz', batch_size=BATCH_SIZE)
loader = LoadData('ecg-two-classes.npz', batch_size=BATCH_SIZE)

signal_length = loader.X.shape[-1]
data_loader = loader.get_loader(ToTensorDataset)
X, y = loader.get_all_data()

print(f"Signal length: {signal_length}")
print(f"Size of the dataset: {len(loader.X)}")
print()

model = Autoencoder(signal_length)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# zanka učenja
print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, _ in data_loader:
        optimizer.zero_grad()
        X_recon, _ = model(X_batch)
        loss = criterion(X_recon, X_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(data_loader):.4f}")

# vrednotenje: napaka rekonstrukcije na vseh podatkih
print("\nEvaluating reconstruction on all data...")
model.eval()
with torch.no_grad():
    X_recon, embeddings = model(X)
    mse = nn.functional.mse_loss(X_recon, X).item()
print(f"Reconstruction MSE: {mse:.6f}")

# nariši 2d vtične vrednosti označene po razredih
embeddings = embeddings.numpy()
y_np = y.numpy()
plt.figure(figsize=(6, 3))
for c in np.unique(y_np):
    plt.scatter(embeddings[y_np == c, 0], embeddings[y_np == c, 1], label=f"Razred {c}", alpha=0.6)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("avtoenk-vložitev.svg")
plt.show()
print("2D graf vektorske vložitve shranjen v 'avtoenk-vložitev.svg'")

# izberi naključni vhodni signal in prikaži rekonstrukcijo
idx = 502
X_random = X[idx:idx+1]  # oblika [1, 1, signal_length]
with torch.no_grad():
    X_recon, _ = model(X_random)
X_orig = X_random.squeeze().numpy()
X_recon = X_recon.squeeze().numpy()

plt.figure(figsize=(10, 4))
plt.plot(X_orig, label='original')
plt.plot(X_recon, label='rekonstruiran')
plt.legend()
plt.title(f"Rekonstrukcija EKG signala (vzorec {idx})")
plt.ylabel("Amplituda")
plt.savefig("avtoenk-rek.svg")
plt.show()
plt.close()

