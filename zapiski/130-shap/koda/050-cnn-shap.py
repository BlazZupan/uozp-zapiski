import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import shap
import matplotlib.pyplot as plt

# parametri
BATCH_SIZE = 32  # velikost podatkov pri paketnem učenju
EPOCHS = 30  # število iteracij učenja
LEARNING_RATE = 0.005  # stopnja učenja

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

        # razdelitev podatkov na učno in testno množico
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # določitev števila razredov
        self.num_classes = len(np.unique(y))

    def get_train_loader(self, ToTensorDataset):
        train_dataset = ToTensorDataset(self.X_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        return train_loader

    def get_test_data(self):
        X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32)
        if X_test_tensor.ndim == 2:
            X_test_tensor = X_test_tensor.unsqueeze(1)
        y_test_tensor = torch.tensor(self.y_test, dtype=torch.long)
        return X_test_tensor, y_test_tensor

    def get_num_classes(self):
        return self.num_classes

# cnn model
class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_layers = nn.Sequential(
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
        
        # popolnoma povezane plasti
        self.fc = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# nalaganje podatkov
loader = LoadData('ecg-two-classes.npz', batch_size=BATCH_SIZE)
num_classes = loader.get_num_classes()
signal_length = loader.X_train.shape[-1]
train_loader = loader.get_train_loader(ToTensorDataset)

print(f"Number of classes: {num_classes}")
print(f"Signal length: {signal_length}")
print(f"Size of the training set: {len(loader.X_train)}")
print(f"Size of the test set: {len(loader.X_test)}")
print()

model = Model(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# zanka učenja
print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
    
    train_accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}, "
          f"Train Accuracy: {train_accuracy:.2f}%")

# vrednotenje
print("\nEvaluating on test set...")
model.eval()
with torch.no_grad():
    X_test_tensor, y_test_tensor = loader.get_test_data()
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test_tensor).float().mean().item()
print(f"Test accuracy: {accuracy:.4f}")
print(f"Test loss: {loss:.4f}")
print("Confusion matrix:")
print(confusion_matrix(y_test_tensor.numpy(), predicted.numpy()))

# After the confusion matrix, add SHAP analysis

# Prepare background data for SHAP
background_data = loader.X_train[:100]  # Use first 100 training samples as background
background_data = torch.tensor(background_data, dtype=torch.float32)
if background_data.ndim == 2:
    background_data = background_data.unsqueeze(1)

# Create SHAP explainer
explainer = shap.DeepExplainer(model, background_data)

# Select 5 random training samples
np.random.seed(42)  # for reproducibility
random_indices = np.random.choice(len(loader.X_train), 5, replace=False)
test_samples = torch.tensor(loader.X_train[random_indices], dtype=torch.float32)
if test_samples.ndim == 2:
    test_samples = test_samples.unsqueeze(1)

# Calculate SHAP values
shap_values = explainer.shap_values(test_samples)
print("SHAP values shape:", [sv.shape for sv in shap_values])
print("Test samples shape:", test_samples.shape)

# Plot the results
plt.figure(figsize=(15, 10))

# Plot each signal with its SHAP values
for i in range(5):
    plt.subplot(5, 1, i+1)
    
    # Plot the original signal
    signal = test_samples[i, 0].numpy()
    x_coords = np.arange(len(signal))
    plt.plot(x_coords, signal, 'b-', alpha=0.5, label='Signal')
    
    # Plot SHAP values as a heatmap
    shap_vals = shap_values[0][0, :, 0]  # Removed i from indexing since first dim is 1
    plt.scatter(x_coords, signal,
               c=shap_vals, cmap='RdBu', 
               s=50, alpha=0.7)
    
    plt.colorbar(label='SHAP value')
    plt.title(f'Sample {i+1}')
    plt.legend()

plt.tight_layout()
# Save the figure BEFORE showing it
plt.savefig('shap-cnn.svg', bbox_inches='tight', pad_inches=0.1)
# Show the plot AFTER saving
plt.show()

