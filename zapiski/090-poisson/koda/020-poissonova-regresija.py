from mgrad_seven import Value
import random
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

class LinReg:
    def __init__(self, n_inputs):
        self.ws = [Value(random.uniform(-1,1), label=f'w{i}') 
                   for i in range(n_inputs)]
        self.b = Value(0.0, label='b')
    
    def __call__(self, x):
        # x je vektor
        return sum(w * xi for w, xi in zip(self.ws, x)) + self.b
    
    def parameters(self):
        return self.ws + [self.b]
    
    def __repr__(self):
        s = ', '.join(f'w{i}={w.data:.3f}' for i, w in enumerate(self.ws))
        return f"LinReg({s}, b={self.b.data:.3f})"
    
    def loss(self, X, ys):
        # X je matrika, v vrsticah so vektorji, ys je vektor
        yhats = [self(x) for x in X]
        return sum([(y - yhat)**2 for y, yhat in zip(ys, yhats)]) / len(ys)
    
    def batch_loss(self, X, ys, m=10):
        indices = random.sample(range(len(X)), m)
        batch_X = [X[idx] for idx in indices]
        batch_ys = [ys[idx] for idx in indices]
        return self.loss(batch_X, batch_ys)
    
class PoissonReg(LinReg):
    def __call__(self, x):
        return super().__call__(x).exp()
    
    def loss(self, X, ys):
        yhats = [self(x) for x in X]
        # negativno log verjetje: -sum(y * log(λ) - λ - log(y!))
        # izpustimo log(y!) ker ta ni odvisen od parametrov
        return -sum([y * yhat.log() - yhat for y, yhat in zip(ys, yhats)]) / len(ys)
    
def train(model, X, ys, learning_rate=0.001, n_epochs=1000, batch_size=10):
    for k in range(n_epochs):
        # izračunamo napovedi in iz njih izgubo
        loss = model.batch_loss(X, ys, m=batch_size)

        # izračunamo gradiente
        loss.backward()

        # osvežimo parametre
        for p in model.parameters():
            p.data -= learning_rate * p.grad
        
        if k % 50 == 0:
            print(f"{k:3} Loss: {model.loss(X, ys).data:5.3f} {model}")
    return model

# preberi podatke iz vhodne datoteke
df = pd.read_excel('data.xlsx')
X = df.iloc[:, :-1].values  # vsi stolpci razen zadnjega
ys = df.iloc[:, -1].values  # zadnji stolpec

# razdeli podatke na učne in testne
X_train, X_test, ys_train, ys_test = train_test_split(X, ys, test_size=0.5, random_state=42)

model = LinReg(n_inputs=X.shape[1])
# model = PoissonReg(n_inputs=X.shape[1])

# učenje modela
model = train(model, X_train, ys_train, n_epochs=1000, batch_size=20, learning_rate=0.01)
print(model)

# poročaj o napaki na testni množici
mae = sum([abs(y - model(x).data) for x, y in zip(X_test, ys_test)]) / len(X_test)
print(f"MAE on test set: {mae:.3f}")

# izračunaj srednjo vrednost in standardni odklon y_test
y_test_mean = np.mean(ys_test)
y_test_std = np.std(ys_test)

print(f"Srednja vrednost y_test: {y_test_mean:.3f}")
print(f"Standardni odklon y_test: {y_test_std:.3f}")

# izračunaj MAE na testni množici, normalizirano na srednjo vrednost in standardni odklon
mae_normalized = mae / y_test_std
print(f"MAE na testni množici, normalizirano na srednjo vrednost in standardni odklon: {mae_normalized:.3f}")
