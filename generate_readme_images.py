import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import shutil

# Import our modules
from chaotic_systems import rossler_system, simulate_system
from embedding import (time_delay_embedding, find_optimal_delay, 
                      estimate_correlation_dimension, mutual_information)

# Create docs/images directory if it doesn't exist
os.makedirs('docs/images', exist_ok=True)

# Set default parameters for Rössler system
params = {'a': 0.2, 'b': 0.2, 'c': 5.7}
initial_state = [1.0, 1.0, 1.0]
t_span = [0, 200]
dt = 0.02
discard_transient = 50

# Run the simulation
print("Simulating Rössler system...")
results = simulate_system(
    rossler_system, 
    initial_state, 
    t_span, 
    dt, 
    params,
    discard_transient
)

time = results['time']
states = results['states']

# Generate 3D plot of the Rössler attractor
print("Generating Rössler 3D plot...")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(states[:, 0], states[:, 1], states[:, 2], lw=0.5, color='blue')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
ax.set_title('Rössler Attractor', fontsize=16)
ax.view_init(elev=30, azim=45)  # Set view angle
plt.tight_layout()
plt.savefig('docs/images/rossler_3d.png', dpi=300, bbox_inches='tight')
plt.close()

# Generate 2D phase space projection
print("Generating phase space plot...")
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(states[:, 0], states[:, 1], lw=0.5, color='blue')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title('Phase Space Projection: x-y plane', fontsize=16)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('docs/images/phase_space.png', dpi=300, bbox_inches='tight')
plt.close()

# Perform embedding on the x component
print("Calculating optimal delay...")
time_series = states[:, 0]
delays = np.arange(1, 100)
mi_values = np.zeros_like(delays, dtype=float)

for i, d in enumerate(delays):
    x = time_series[:-d] if d > 0 else time_series
    y = time_series[d:] if d > 0 else time_series
    mi_values[i] = mutual_information(x, y)

optimal_delay = find_optimal_delay(time_series, 100)
print(f"Optimal delay: {optimal_delay}")

# Plot mutual information
print("Generating mutual information plot...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(delays, mi_values)
ax.axvline(x=optimal_delay, color='r', linestyle='--', 
           label=f'Optimal delay: {optimal_delay}')
ax.set_xlabel('Delay')
ax.set_ylabel('Mutual Information')
ax.set_title('Mutual Information vs Delay', fontsize=16)
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('docs/images/mutual_information.png', dpi=300, bbox_inches='tight')
plt.close()

# Perform time-delay embedding
print("Performing time-delay embedding...")
embedding_dim = 3
embedded_data = time_delay_embedding(time_series, embedding_dim, optimal_delay)

# Plot embedded attractor
print("Generating embedded attractor plot...")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(embedded_data[:, 0], embedded_data[:, 1], embedded_data[:, 2], lw=0.5, color='blue')
ax.set_xlabel(f'$x(t)$')
ax.set_ylabel(f'$x(t+{optimal_delay})$')
ax.set_zlabel(f'$x(t+{2*optimal_delay})$')
ax.set_title(f'Reconstructed Attractor (τ={optimal_delay}, m={embedding_dim})', fontsize=16)
ax.view_init(elev=30, azim=45)  # Set view angle
plt.tight_layout()
plt.savefig('docs/images/embedded_attractor.png', dpi=300, bbox_inches='tight')
plt.close()

# Estimate correlation dimension
print("Estimating correlation dimension...")
dim_results = estimate_correlation_dimension(embedded_data, num_scales=20)
dimension = dim_results['dimension']
print(f"Estimated correlation dimension: {dimension:.3f}")

# Plot correlation dimension
print("Generating dimension estimation plot...")
fig, ax = plt.subplots(figsize=(10, 6))
        
# Plot all points
ax.scatter(dim_results['log_eps'], dim_results['log_c'], 
          s=30, alpha=0.7, label='All points')

# Highlight scaling region
scaling_region = dim_results['scaling_region']
ax.scatter(dim_results['log_eps'][scaling_region[0]:scaling_region[1]], 
          dim_results['log_c'][scaling_region[0]:scaling_region[1]], 
          s=50, color='red', label='Scaling region')

# Plot the fit line
x_fit = np.linspace(dim_results['scaling_log_eps'][0], 
                  dim_results['scaling_log_eps'][-1], 100)
y_fit = dim_results['fit_slope'] * x_fit + dim_results['fit_intercept']
ax.plot(x_fit, y_fit, 'k--', 
       label=f'Fit: D = {dim_results["dimension"]:.3f}')

ax.set_xlabel('log(ε)')
ax.set_ylabel('log(C(ε))')
ax.set_title(f'Correlation Dimension Estimation (m={embedding_dim})', fontsize=16)
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('docs/images/dimension_estimation.png', dpi=300, bbox_inches='tight')
plt.close()

print("All example images generated successfully!")
print("Image files are in the 'docs/images' directory.")

# List generated images
image_files = os.listdir('docs/images')
print("\nGenerated images:")
for image in image_files:
    print(f" - docs/images/{image}") 