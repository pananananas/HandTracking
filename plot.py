import matplotlib.pyplot as plt
import numpy as np

ax = plt.axes(projection='3d')

x_data = np.arange(0, 4*np.pi, 0.1)
y_data = np.arange(0, 4*np.pi, 0.1)

X, Y = np.meshgrid(x_data, y_data)
Z = np.sin(X) + np.cos(Y)

ax.plot_surface(X, Y, Z, cmap='plasma', edgecolor='none')
ax.view_init(azim=0, elev=90)
plt.show()