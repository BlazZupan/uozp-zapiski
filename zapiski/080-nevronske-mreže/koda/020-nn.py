import random
from mgrad_six import Value
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
import math
from mplot import plot_decision_boundary

class Neuron:

    def __init__(self, nin, activation='relu'):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.activation = activation
        self.out = None

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        out = act.sigmoid() if self.activation == 'sigmoid' else act.relu()
        self.out = out
        return out

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"Neuron({len(self.w)})"

class Layer:

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class NeuralNetwork:

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], activation="sigmoid" if i == len(nouts)-1 else "relu") for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"NN of [{', '.join(str(layer) for layer in self.layers)}]"
    
    def loss(self, X, ys):
        yhats = [self(x) for x in X]
        return -sum(y * yhat.log() + (1-y) * (1-yhat).log() for y, yhat in zip(ys, yhats)) / len(ys)
    
    def batch_loss(self, X, ys, m=20):
        indices = random.sample(range(len(X)), m)
        batch_X = [X[idx] for idx in indices]
        batch_ys = [ys[idx] for idx in indices]
        return self.loss(batch_X, batch_ys)

# generiraj sintetične podatke za učno množico
random.seed(0)
n = 300
from sklearn.datasets import make_moons, make_blobs
X, ys = make_moons(n_samples=n, noise=0.1)

# izriši podatke
X0 = [x for x, y in zip(X, ys) if y == 0]
X1 = [x for x, y in zip(X, ys) if y == 1]

plt.figure(figsize=(10, 8))
plt.scatter([x[0] for x in X0], [x[1] for x in X0], c='blue', label='y=0')
plt.scatter([x[0] for x in X1], [x[1] for x in X1], c='red', label='y=1')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Podatki z nelinearno ločljivim razredom')
plt.savefig("lunice.svg")

random.seed(10)
n_hidden = [6, 2]
model = NeuralNetwork(2, n_hidden + [1])

max_iter = 2000
prev_loss = float('inf')

for k in range(max_iter):
    loss = model.batch_loss(X, ys)  # izračunaj izgubo
    loss.backward()  # izračunaj gradiente

    # stopnja učenja se zmanjšuje s časom
    learning_rate = 0.1 - 0.09*k/max_iter 
    for p in model.parameters():
        p.data -= learning_rate * p.grad
    
    if k % 100 == 0:
        current_loss = model.loss(X, ys).data
        print(f"{k:4} Full loss: {current_loss:5.3f}")        
        prev_loss = current_loss

print(f"Final model: {model}")

fig = plot_decision_boundary(model, X, ys, grid_size=50)
fig.savefig("odločitvena-meja.svg")


# for all the data points, compute the activation of the penultimate layer, and plot the data points in the space of the activations
o1, o2 = [], []
for x in X:
    model(x)  # forward pass to compute activations
    o1.append(model.layers[1].neurons[0].out.data)
    o2.append(model.layers[1].neurons[1].out.data)
plt.figure(figsize=(10, 8))
plt.scatter(o1, o2, c=['blue' if y == 0 else 'orange' for y in ys], alpha=0.5)
plt.xlabel('Aktivacija prvega nevrona v zadnjem skritem nivoju')
plt.ylabel('Aktivacija drugega nevrona v zadnjem skritem nivoju')
plt.savefig("aktivacijski-prostor.svg")

def plot_neuron_activation(model, X, neuron=0):
    plt.figure(figsize=(10, 8))
    # Get activations for all data instances
    activations = []
    for x in X:
        model(x)
        activations.append(model.layers[0].neurons[neuron].out.data)
    activations = np.array(activations)
    # scale the activations to be between 0 and 1
    activations = (activations - np.min(activations)) / (np.max(activations) - np.min(activations)) + 1
    # Plot with size and color proportional to activation
    scatter = plt.scatter(X[:, 0], X[:, 1], c=activations, cmap='coolwarm', 
                        alpha=0.5, s=activations * 100)
    plt.colorbar(scatter, label=f'Aktivacija nevrona {neuron} v prvem skritem nivoju')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True)
    plt.savefig(f"aktivacija-nevrona.svg")

# Compute, store and report the mean activations for a data sample
n_samples = 50
sample_indices = np.random.choice(len(X), n_samples, replace=False)
sample_X = X[sample_indices]

activations = np.zeros((n_samples, n_hidden[0]))  # 8 neurons in layer 0
for i, x in enumerate(sample_X):
    model(x)  # forward pass
    for j in range(n_hidden[0]):  # for each neuron in layer 0
        activations[i, j] = model.layers[0].neurons[j].out.data

print(f"\nMean activations in the first hidden layer (based on {n_samples} samples):")
for i in range(n_hidden[0]):
    print(f"Neuron {i}: {np.mean(activations[:, i]):.4f}")


