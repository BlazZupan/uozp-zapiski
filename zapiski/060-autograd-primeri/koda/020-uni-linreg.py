from mgrad_four import Value
from mplot import draw_dot
import random
import matplotlib.pyplot as plt

class LinReg:
    def __init__(self):
        self.w = Value(random.uniform(-1,1), label='w')
        self.b = Value(0.0, label='b')
    
    def __call__(self, x):
        return self.w * x + self.b
    
    def parameters(self):
        return [self.w, self.b]

    def loss(self, xs, ys):
        yhats = [self(x) for x in xs]
        return sum([(y - yhat)**2 for y, yhat in zip(ys, yhats)]) / len(xs)
    
    def batch_loss(self, xs, ys, m=10):
        indices = random.sample(range(len(xs)), m)
        batch_xs = [xs[idx] for idx in indices]
        batch_ys = [ys[idx] for idx in indices]
        return self.loss(batch_xs, batch_ys)
    
    def __repr__(self):
        return f"LinReg(w={self.w.data:.3f}, b={self.b.data:.3f})"
    
# učni podatki
random.seed(42)
n = 1000
xs = [random.uniform(-10, 10) for _ in range(n)]
ys = [2*x + 1 + random.gauss(0, 5) for x in xs]

plt.scatter(xs, ys, c='black', alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('linreg-podatki.svg')
plt.close()

model = LinReg()

def train(model, xs, ys, learning_rate=0.001, n_epochs=1000, batch_size=10):
    for k in range(n_epochs):
        # izračunamo napovedi in iz njih izgubo
        loss = model.batch_loss(xs, ys, batch_size)

        # izračunamo gradiente
        loss.backward()

        # osvežimo parametre
        for p in model.parameters():
            p.data -= learning_rate * p.grad
        
        if k % 100 == 0:
            print(f"{k:3} Loss: {model.loss(xs, ys).data:5.3f} {model}")
    return model

model = train(model, xs, ys, n_epochs=10000, batch_size=20)
print(model)
