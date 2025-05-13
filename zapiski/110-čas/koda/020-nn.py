import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

BATCH_SIZE = 32  # velikost podatkov pri paketnem učenju
EPOCHS = 30  # število iteracij učenja
LEARNING_RATE = 0.005  # stopnja učenja

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

# model s popolnoma povezanimi plastmi
class Model(nn.Module):
    def __init__(self, num_classes, signal_length):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(signal_length, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # oblika x: [serija, 1, dolžina_signala] ali [serija, dolžina_signala]
        x = x.view(x.size(0), -1)  # splošči na [serija, dolžina_signala]
        return self.fc(x)

# nalaganje podatkov
# loader = LoadData('ecg-two-classes.npz', batch_size=BATCH_SIZE)
loader = LoadData('ecg-four-classes.npz', batch_size=BATCH_SIZE)
num_classes = loader.get_num_classes()
signal_length = loader.X_train.shape[-1]
train_loader = loader.get_train_loader(ToTensorDataset)

print(f"Number of classes: {num_classes}")
print(f"Signal length: {signal_length}")
print(f"Size of the training set: {len(loader.X_train)}")
print(f"Size of the test set: {len(loader.X_test)}")
print()

model = Model(num_classes, signal_length)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# zanka učenja
print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
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

# shrani model
torch.save(model.state_dict(), 'ecg_cnn_model.pth')
print("Model saved as 'ecg_cnn_model.pth'")

# poročaj o številu parametrov v vsaki plasti
print("\nNumber of parameters in each layer:")
total_params = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        num_params = param.numel()
        print(f"{name}: {num_params:,}")
        total_params += num_params
print(f"Total trainable parameters: {total_params:,}")
