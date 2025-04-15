from mgrad_four import Value
import random
import matplotlib.pyplot as plt

class MDS:
    def __init__(self, items):
        self.pos = {i: [Value(random.uniform(-1, 1)) for _ in range(2)] 
                    for i in items}

    def __call__(self, x):
        # vrne koordinate mesta x
        return self.pos[x]
    
    def parameters(self):
        return [p for pos in self.pos.values() for p in pos]
    
    def d(self, a, b):
        # vrne evklidsko razdaljo med mestoma a in b
        return sum([(ai - bi) ** 2 for ai, bi in zip(self.pos[a], self.pos[b])]) ** 0.5

    def loss(self, data):
        return sum((self.d(*pair) - data[pair])**2 
                   for pair in data.keys()) / len(data)

# učni podatki
distances = {
    ("Novo Mesto", "Maribor"): 172,
    ("Novo Mesto", "Celje"): 83,
    ("Novo Mesto", "Koper"): 172,
    ("Novo Mesto", "Kranj"): 102,
    ("Novo Mesto", "Ljubljana"): 72,
    ("Novo Mesto", "Postojna"): 118,
    ("Maribor", "Celje"): 55,
    ("Maribor", "Koper"): 234,
    ("Maribor", "Kranj"): 156,
    ("Maribor", "Ljubljana"): 128,
    ("Maribor", "Postojna"): 180,
    ("Celje", "Koper"): 184,
    ("Celje", "Kranj"): 107,
    ("Celje", "Ljubljana"): 79,
    ("Celje", "Postojna"): 131,
    ("Koper", "Kranj"): 132,
    ("Koper", "Ljubljana"): 107,
    ("Koper", "Postojna"): 60,
    ("Kranj", "Ljubljana"): 33,
    ("Kranj", "Postojna"): 77,
    ("Ljubljana", "Postojna"): 53,
}

items = set(i for pair in distances.keys() for i in pair)

def train(model, learning_rate=0.001, n_epochs=1000):
    for k in range(n_epochs):
        loss = model.loss(distances)  # izračunamo napovedi in iz njih izgubo
        loss.backward()  # izračunamo gradiente

        #  korak gradientnega spusta
        for p in model.parameters():
            p.data -= learning_rate * p.grad
        
        if k % 100 == 0:
            print(f"{k:3} Loss: {model.loss(distances).data:5.3f}")
    return model

random.seed(42)
model = MDS(items)
model = train(model, n_epochs=2000, learning_rate=0.01)

plt.figure(figsize=(5, 5))
for city in items:
    x, y = model(city)
    plt.scatter(x.data, y.data)
    plt.text(x.data, y.data, city)
plt.savefig("mds-mesta.svg")