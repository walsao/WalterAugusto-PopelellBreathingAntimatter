# Walter Augusto Popelell Equation - 2D Breathing Matter + Antimatter Simulation

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set up simulation parameters
L = 100             # Length of space
N = 200             # Grid size (N x N)
dx = L / N          # Space step
dt = 0.005          # Time step
T = 1500            # Number of time steps
gravity_strength = 0.01  # Gentle breathing pullback

learning_rate = 5.0
decay_rate = 0.001

# Create spatial grid
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)

# Initialize fields
phi_m = np.random.randn(N, N) * 0.05   # Matter breathing field
phi_m_old = np.copy(phi_m)

phi_am = np.random.randn(N, N) * 0.05  # Antimatter breathing field
phi_am_old = np.copy(phi_am)

Popelell = np.ones((N, N)) * 5.0        # Learning nonlinearity field

# Add STRONG breathing blobs for matter 💥
phi_m[N//4, N//4] += 10.0
phi_m[3*N//4, 3*N//4] += 10.0

# Add STRONG breathing blobs for antimatter 💥
phi_am[N//4, 3*N//4] += -10.0
phi_am[3*N//4, N//4] += -10.0

# Laplacian operator in 2D
def laplacian(phi, dx):
    return (np.roll(phi, 1, axis=0) + np.roll(phi, -1, axis=0) +
            np.roll(phi, 1, axis=1) + np.roll(phi, -1, axis=1) -
            4 * phi) / dx**2

# Set up plot
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
im_m = axs[0].imshow(np.zeros((N, N)), extent=[0, L, 0, L], origin='lower', cmap='Reds', vmin=-5, vmax=5)
im_am = axs[1].imshow(np.zeros((N, N)), extent=[0, L, 0, L], origin='lower', cmap='Blues', vmin=-5, vmax=5)
im_pop = axs[2].imshow(np.zeros((N, N)), extent=[0, L, 0, L], origin='lower', cmap='plasma', vmin=0, vmax=100)

axs[0].set_title('Matter Breathing Field (phi_m)')
axs[1].set_title('Antimatter Breathing Field (phi_am)')
axs[2].set_title('Popelell Field (Nonlinearity)')
fig.suptitle('Walter Augusto Popelell Equation - 2D Breathing Matter + Antimatter Universe')

plt.tight_layout()

# Update function for animation
def update(frame):
    global phi_m, phi_m_old, phi_am, phi_am_old, Popelell

    # Calculate Laplacians
    lap_m = laplacian(phi_m, dx)
    lap_am = laplacian(phi_am, dx)

    # Estimate breathing energy (total)
    breathing_energy = 0.5 * (lap_m**2 + phi_m**2) + 0.5 * (lap_am**2 + phi_am**2)

    # Update Popelell field
    Popelell += dt * (learning_rate * breathing_energy - decay_rate * Popelell)
    Popelell = np.maximum(Popelell, 0.01)

    # Update matter breathing field (positive nonlinearity)
    phi_m_new = (2 * phi_m - phi_m_old +
                 dt**2 * (lap_m - gravity_strength * phi_m - Popelell * phi_m**3))

    # Update antimatter breathing field (negative nonlinearity)
    phi_am_new = (2 * phi_am - phi_am_old +
                  dt**2 * (lap_am - gravity_strength * phi_am + Popelell * phi_am**3))

    # Breathing Annihilation Effect: where phi_m and phi_am overlap, damp both
    overlap = phi_m * phi_am
    annihilation_zone = np.abs(overlap) > 1.0
    phi_m_new[annihilation_zone] *= 0.5
    phi_am_new[annihilation_zone] *= 0.5

    phi_m_old = np.copy(phi_m)
    phi_am_old = np.copy(phi_am)
    phi_m = np.copy(phi_m_new)
    phi_am = np.copy(phi_am_new)

    # Update plots
    im_m.set_data(phi_m)
    im_am.set_data(phi_am)
    im_pop.set_data(Popelell)

    return im_m, im_am, im_pop

# Animate
ani = animation.FuncAnimation(fig, update, frames=T, interval=30, blit=True)
plt.show()
