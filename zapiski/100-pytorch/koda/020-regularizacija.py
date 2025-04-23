import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split


# naloži podatke
df = pd.read_excel('body-fat-brozek.xlsx')
X = df.iloc[:, :-1].values
X = (X - X.mean(axis=0)) / X.std(axis=0)
ys = df.iloc[:, -1].values

def to_tensor(X):
    return torch.tensor(X, dtype=torch.float32)

# razdeli podatke na učno in testno množico
X_train, X_test, ys_train, ys_test = \
    (to_tensor(d) for d in train_test_split(X, ys, test_size=0.5, random_state=42))
X_train, ys_train = to_tensor(X_train), to_tensor(ys_train).view(-1, 1)
X_test, ys_test = to_tensor(X_test), to_tensor(ys_test).view(-1, 1)

# postavi model in optimizator
model = nn.Linear(X.shape[1], 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
lambda_reg = 0.01

def soft_threshold(param, lmbd):
    with torch.no_grad():
        param.copy_(param.sign() * torch.clamp(param.abs() - lmbd, min=0.0))

# učenje modela
for epoch in range(1000):
    optimizer.zero_grad()
    mse_loss = nn.MSELoss()(model(X_train), ys_train)
    l1_norm = sum(param.abs().sum() for name, param in model.named_parameters() if 'weight' in name)
    loss = mse_loss + lambda_reg * l1_norm
    loss.backward()
    optimizer.step()
    soft_threshold(model.weight, lambda_reg)

    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}/1000, Loss: {loss.item():.4f}")

# ocena modela in pomembnosti značilk
with torch.no_grad():
    print(f"Test MAE: {torch.mean(torch.abs(model(X_test) - ys_test)).item():.4f}")

    # izpiši pomembnosti značilk
    weights = model.weight.data.squeeze().tolist()
    features = df.columns[:-1]
    sorted_weights = sorted(zip(weights, features), 
                          key=lambda x: abs(x[0]), 
                          reverse=True)
    for weight, name in sorted_weights:
        if weight != 0:
            print(f"{name:9s}: {weight:7.4f}")
