
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

# Generate the rectangular grid point cloud
grid_size = 100
x = np.linspace(0, 1, grid_size)
y = np.linspace(0, 1, grid_size)
X, Y = np.meshgrid(x, y)
points = np.stack([X.ravel(), Y.ravel()], axis=1)  # shape (N, 2)

# Gaussian centers
centers = np.array([
    [0.136, 0.863],
    [0.62, 0.62],
    [0.439, 0.237]
])

# Example: Use the same covariance matrix for all
cov = np.array([[0.01, 0], [0, 0.01]])  # isotropic, adjust for spread

# Evaluate Gaussians
values = np.zeros(len(points))
for mu in centers:
    rv = multivariate_normal(mean=mu, cov=cov)
    values += rv.pdf(points)  # sum the densities

# Normalize for display (optional)
values /= values.max()

# Reshape to 2D image for visualization
heatmap = values.reshape(grid_size, grid_size)

# Visualize
plt.imshow(heatmap, extent=(0, 1, 0, 1), origin='lower', cmap='hot')
plt.colorbar(label='Gaussian value')
plt.title("Sum of Discrete Gaussians on Point Cloud")
plt.scatter(centers[:,0], centers[:,1], c='cyan', marker='x')  # show centers
plt.show()