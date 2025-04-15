from mgrad_four import Value
from mplot import draw_dot
import random

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

    
# učni podatki, y = 2*x1 + 3*x2 - x3 + 1 + šum
n = 1000
X = [[random.uniform(-10, 10) for _ in range(3)] for _ in range(n)]
ys = [2*x[0] + 3*x[1] - x[2] + 1 + random.gauss(0, 5) for x in X]

model = LinReg(n_inputs=3)

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

model = train(model, X, ys, n_epochs=1000, batch_size=20, learning_rate=0.01)
print(model)
