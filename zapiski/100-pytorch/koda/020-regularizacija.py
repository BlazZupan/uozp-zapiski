import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split

# Load and prepare data
df = pd.read_excel('body-fat-brozek.xlsx')
X = df.iloc[:, :-1].values
X = (X - X.mean(axis=0)) / X.std(axis=0)
ys = df.iloc[:, -1].values

def to_tensor(X):
    return torch.tensor(X, dtype=torch.float32)

# Split and convert to tensors
X_train, X_test, ys_train, ys_test = \
    (to_tensor(d) for d in train_test_split(X, ys, test_size=0.5, random_state=42))
X_train, ys_train = to_tensor(X_train), to_tensor(ys_train).view(-1, 1)
X_test, ys_test = to_tensor(X_test), to_tensor(ys_test).view(-1, 1)

# Training setup
model = nn.Linear(X.shape[1], 1)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
lambda_reg = 0.5

def soft_threshold(param, lmbd):
    with torch.no_grad():
        param.copy_(param.sign() * torch.clamp(param.abs() - lmbd, min=0.0))

# Training loop
for epoch in range(1000):
    optimizer.zero_grad()
    mse_loss = nn.MSELoss()(model(X_train), ys_train)
    l1_norm = sum(param.abs().sum() for name, param in model.named_parameters() if 'weight' in name)
    loss = mse_loss + lambda_reg * l1_norm
    loss.backward()
    optimizer.step()
    soft_threshold(model.weight, lambda_reg * 0.1)

    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}/1000, Loss: {loss.item():.4f}")

# Evaluation and feature importance
with torch.no_grad():
    print(f"Test MAE: {torch.mean(torch.abs(model(X_test) - ys_test)).item():.4f}")
    for weight, name in sorted(zip(model.weight.data.squeeze().tolist(), df.columns[:-1]), key=lambda x: abs(x[0]), reverse=True):
        print(f"{name}: {weight:.4f}")
