import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# učni podatki, ki so primerni za Poissonovo regresijo
# lambda = exp(2*x1 + 3*x2 - x3 + 1)
n = 1000
X = [[random.uniform(-1, 1) for _ in range(3)] for _ in range(n)]
ys = [np.random.poisson(np.exp(2*x[0] + 3*x[1] - x[2] + 1)) for x in X]

# plot distribution of ys
plt.hist(ys, bins=30)
plt.xlabel("y")
plt.ylabel("Frequency")
plt.savefig('podatki-porazdelitev-y.svg')

# save data to excel, first column is x1, second is x2, third is x3, fourth is y
df = pd.DataFrame(X, columns=['x1', 'x2', 'x3'])
df['y'] = ys
df.to_excel('data.xlsx', index=False)

