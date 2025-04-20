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
X_train, X_test, ys_train, ys_test = (
    to_tensor(d) for d in train_test_split(X, ys, test_size=0.5, 
                                         random_state=42))
X_train = to_tensor(X_train)
ys_train = to_tensor(ys_train).view(-1, 1)
X_test = to_tensor(X_test)
ys_test = to_tensor(ys_test).view(-1, 1)

# Training setup
model = nn.Linear(X.shape[1], 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Training loop
for epoch in range(10000):
    optimizer.zero_grad()
    loss = nn.MSELoss()(model(X_train), ys_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 1000 == 0:
        print(f"Epoch {epoch+1}/1000, Loss: {loss.item():.4f}")

# Evaluation and feature importance
with torch.no_grad():
    mae = torch.mean(torch.abs(model(X_test) - ys_test)).item()
    print(f"Test MAE: {mae:.4f}")
    
    weights = model.weight.data.squeeze().tolist()
    features = df.columns[:-1]
    sorted_weights = sorted(zip(weights, features), 
                          key=lambda x: abs(x[0]), 
                          reverse=True)
    for weight, name in sorted_weights:
        print(f"{name}: {weight:.4f}")
