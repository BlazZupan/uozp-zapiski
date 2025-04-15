from mgrad_three import Value
from mplot import draw_dot
import random

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
        return sum([(y - yhat)**2  for y, yhat in zip(ys, yhats)])
    
    def __repr__(self):
        return f"LinReg(w={self.w.data:.3f}, b={self.b.data:.3f})"
    
# učni podatki
xs = [2., 3., 3.5, 5., -2., -5.]
ys = [2*x + 1 for x in xs]

# začnemo z nekaj poskusi
random.seed(42)
model = LinReg()
print(model)
loss = model.loss(xs, ys)
print(loss)

graf = draw_dot(loss)
graf.render('linreg-izguba', format='svg', cleanup=True)

# učimo model

model = LinReg()

def train(model, xs, ys, learning_rate=0.001, n_epochs=1000):
    for k in range(n_epochs):
        # izračunamo napovedi in iz njih izgubo
        loss = model.loss(xs, ys)

        # izračunamo gradiente
        loss.backward()

        # osvežimo parametre
        for p in model.parameters():
            p.data -= learning_rate * p.grad
        
        if k % 50 == 0:
            print(f"{k:3} Loss: {loss.data:5.3f} {model}")
    return model

model = train(model, xs, ys, n_epochs=500)
print(model)
