import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_excel('body-fat-brozek.xlsx')

# pretty print the names input features (all but the last column), separated by commas
print(f'Input features ({len(df.columns[:-1])}): {", ".join(df.columns[:-1])}')
print(f'Output feature ({len(df.columns[-1])}): {df.columns[-1]}')

# show the distribution of the output feature, plot it in a histogram
plt.hist(df.iloc[:, -1], bins=10, edgecolor='black')
plt.xlabel('Delež maščob v telesu')
plt.ylabel('Frekvenca')
plt.savefig('body-fat-histogram.svg')
plt.close()

# compute mean, standard deviation of the output feature
mean = df.iloc[:, -1].mean()
std = df.iloc[:, -1].std()
print(f'Mean: {mean:.2f}, Standard deviation: {std:.2f}')