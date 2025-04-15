import numpy as np
import matplotlib.pyplot as plt
import math

def plot_decision_boundary(model, X, ys, grid_size=20):
    # Compute axis limits with padding and round to whole numbers
    X = np.array(X)
    X0 = [x for x, y in zip(X, ys) if y == 0]
    X1 = [x for x, y in zip(X, ys) if y == 1]

    x_min, x_max = X[:,0].min(), X[:,0].max()
    y_min, y_max = X[:,1].min(), X[:,1].max()
    padding = 0.2
    x_min, x_max = math.floor(x_min - padding * (x_max - x_min)), math.ceil(x_max + padding * (x_max - x_min))
    y_min, y_max = math.floor(y_min - padding * (y_max - y_min)), math.ceil(y_max + padding * (y_max - y_min))

    # Create a mesh grid for visualization
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                        np.linspace(y_min, y_max, grid_size))

    # Calculate predictions for the mesh grid
    Z = np.array([[model([x, y]).data for x, y in zip(x_row, y_row)]
                for x_row, y_row in zip(xx, yy)])

    # Plot the decision boundary and data points
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.8)
    plt.colorbar(label='Verjetnost razreda 1')
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)

    # Plot the data points
    plt.scatter([x[0] for x in X0], [x[1] for x in X0], c='blue', label='y=0', alpha=0.5)
    plt.scatter([x[0] for x in X1], [x[1] for x in X1], c='orange', label='y=1', alpha=0.5)

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Odločitvena meja')
    plt.legend()
    plt.grid(True)
    return plt