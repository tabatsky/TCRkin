import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm

# Define the grid
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)

# Define the two surfaces
def z1(x, y):
    return x-x  # Lower surface

def z2(x, y):
    return x+y+np.sqrt(4*x*y)  # Upper surface

Z1 = z1(X, Y)
Z2 = z2(X, Y)

# Create the figure
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the two surfaces
ax.plot_surface(X, Y, Z1, color='blue', alpha=0.2, label='z1(x,y)')
ax.plot_surface(X, Y, Z2, color='blue', alpha=0.2, label='z2(x,y)')

# ===== PROPERLY FILL THE VOLUME BETWEEN SURFACES =====
# Create side walls to enclose the volume
verts = []

# Create side walls along x-boundaries
for i in [0, -1]:
    for j in range(len(y)-1):
        # Wall segment connecting lower and upper surfaces
        verts.append([
            (x[i], y[j], Z1[j,i]),
            (x[i], y[j+1], Z1[j+1,i]),
            (x[i], y[j+1], Z2[j+1,i]),
            (x[i], y[j], Z2[j,i])
        ])

# Create side walls along y-boundaries
for j in [0, -1]:
    for i in range(len(x)-1):
        # Wall segment connecting lower and upper surfaces
        verts.append([
            (x[i], y[j], Z1[j,i]),
            (x[i+1], y[j], Z1[j,i+1]),
            (x[i+1], y[j], Z2[j,i+1]),
            (x[i], y[j], Z2[j,i])
        ])

# Create Poly3DCollection for the walls
walls = Poly3DCollection(verts, alpha=0.6, facecolor='blue')
ax.add_collection3d(walls)

# Fill the top and bottom
ax.plot_surface(X, Y, Z1, color='blue', alpha=0.8)
ax.plot_surface(X, Y, Z2, color='blue', alpha=0.8)

# Plot upper surface with colormap and mesh
"""
upper_surf = ax.plot_surface(X, Y, Z2, cmap=cm.coolwarm, alpha=0.7,
                           edgecolor='black', linewidth=0.3, 
                           label='Upper surface')
"""
upper_surf = ax.plot_surface(X, Y, Z2, cmap=cm.Blues, alpha=0.6,
                           edgecolor='black', linewidth=0.3, 
                           label='Upper surface')

# ===== END VOLUME FILLING =====

# Add labels and title
label_params = {'fontsize': 26, 'fontweight': 'normal', 'fontstyle': 'italic'}
ax.set_xlabel('Kon*L, 1/min', **label_params, labelpad=30)
ax.set_ylabel('Koff, 1/min', **label_params, labelpad=30)
ax.set_zlabel('Krel, 1/min', **label_params, labelpad=30)
#ax.set_title('Properly Filled 3D Volume Between Two Surfaces')

# Increase tick label size for all axes
ax.tick_params(axis='x', labelsize=21, pad=8)
ax.tick_params(axis='y', labelsize=21, pad=8)
ax.tick_params(axis='z', labelsize=21, pad=12)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    #Patch(facecolor='blue', alpha=0.8, label='Lower Surface z1(x,y)'),
    #Patch(facecolor='blue', alpha=0.6, label='Upper Boundary'),
    Patch(facecolor='blue', alpha=0.8, label='Hump area')
]

# Position legend in upper right with precise bbox_to_anchor
legend = ax.legend(handles=legend_elements, 
                  loc='upper right',
                  bbox_to_anchor=(1.0, 1.0),  # Adjusted to upper right corner
                  fontsize=21,
                  framealpha=1,
                  borderaxespad=0.6)



# Adjust viewing angle
#ax.view_init(elev=20, azim=65)
ax.view_init(elev=15, azim=-160)

plt.tight_layout()
plt.show()