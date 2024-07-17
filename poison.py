import numpy as np
import matplotlib.pyplot as plt

N = 64
M = 64
h = 2/64

# Initialize the grid
y = np.zeros([N+2, M+2])

# Set boundary conditions
y[0, :] = 0
y[N+1, :] = 0
y[:, 0] = 0
y[:, M+1] = 0

yp = y.copy()

# Setup the source term
f = np.zeros([N+2, M+2])
xc = np.linspace(-1, 1, N+2)
yc = np.linspace(-1, 1, M+2)
xv, yv = np.meshgrid(xc, yc)
f = np.exp(-20*(xv**2 + yv**2))
print(f)
eps = 1e-6
error = 1

# Jacobi method
while error > eps:
    yp[1:N+1, 1:M+1] = (y[0:N, 1:M+1] + y[2:N+2, 1:M+1] + y[1:N+1, 0:M] + y[1:N+1, 2:M+2]) / 4
    yp[1:N+1, 1:M+1] = yp[1:N+1, 1:M+1] + h**2 * f[1:N+1, 1:M+1]
    error = np.max(np.absolute(yp[1:N+1, 1:M+1] - y[1:N+1, 1:M+1]))
    y[:, :] = yp[:, :]
print(y)
# Plotting the potential
plt.imshow(y, cmap='viridis', extent=[-1, 1, -1, 1])
plt.colorbar(label='Potential')
plt.title('Potential Distribution')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

