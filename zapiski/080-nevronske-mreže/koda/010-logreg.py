import random
import matplotlib.pyplot as plt
from mgrad_six import Value
import mplot

class LogReg:
    def __init__(self, n_inputs):
        self.weights = [Value(random.uniform(-1,1), label=f'w{i}') for i in range(n_inputs)]
        self.b = Value(0.0, label='b')
    
    def __call__(self, x):
        # linearna kombinacija vhodov
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.b
        # sigmoidna aktivacijska funkcija
        return z.sigmoid()
    
    def parameters(self):
        return self.weights + [self.b]
    
    def __repr__(self):
        weights_str = ', '.join(f'w{i}={w.data:.3f}' for i, w in enumerate(self.weights))
        return f"LogReg({weights_str}, b={self.b.data:.3f})"
    
    def loss(self, X, ys):
        # izguba za binarno klasifikacijo (binary cross-entropy)
        yhats = [self(x) for x in X]
        return -sum(y * yhat.log() + (1-y) * (1-yhat).log() for y, yhat in zip(ys, yhats)) / len(ys)
    
    def batch_loss(self, X, ys, m=10):
        indices = random.sample(range(len(X)), m)
        batch_X = [X[idx] for idx in indices]
        batch_ys = [ys[idx] for idx in indices]
        return self.loss(batch_X, batch_ys)

n = 500
from sklearn.datasets import make_moons, make_blobs
X, ys = make_moons(n_samples=n, noise=0.1)
X0 = [x for x, y in zip(X, ys) if y == 0]
X1 = [x for x, y in zip(X, ys) if y == 1]

model = LogReg(2)
print(model(X[0]))

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
            print(f"{k:3} Loss: {model.loss(X, ys).data:5.3f} {model")
    return model

model = train(model, X, ys, n_epochs=2000, batch_size=20, learning_rate=0.03)
print(model)
print("Izguba:", model.loss(X, ys).data)
fig = mplot.plot_decision_boundary(model, X, ys, grid_size=100)
fig.savefig("logreg-meja.svg")