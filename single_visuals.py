# autopep8: off
#%% Imports
import os
import numpy as np
import matplotlib.pyplot as plt

#%% Load files

NAME_DIR = "20251213_174501_CFL_0.2_nite_200_local_False_alpha_4_Minf_0.273"
PATH_DIR = os.path.dirname(os.path.realpath(__file__)) + f"/results/{NAME_DIR}/"

# Load data
convergence_data = np.load(PATH_DIR + "convergence_data.npz")
surf_data = np.loadtxt(PATH_DIR + "surf.csv", delimiter=",", skiprows=1)

# Extract simulation parameters
nite = int(convergence_data['nite'])
CFL = float(convergence_data['CFL'])
Minf = float(convergence_data['Minf'])
alpha = float(convergence_data['alpha'])
localTimeStep = bool(convergence_data['localTimeStep'])

# Extract convergence histories
residus_history = convergence_data['residus_history']
diff_history = convergence_data['diff_history']

#%% Plot convergence data - Residuals
plt.figure(figsize=(10, 6))
plt.semilogy(residus_history[:, 0], label='Residu rho')
plt.semilogy(residus_history[:, 1], label='Residu rhou')
plt.semilogy(residus_history[:, 2], label='Residu rhov')
plt.xlabel('Iteration')
plt.ylabel('Résidu (norme L2)')
plt.title(f'Convergence de la simulation CFD\n(Mach = {Minf}, Alpha = {alpha}°, CFL = {CFL})')
plt.legend()
plt.grid()
plt.show()

# Plot convergence data - Differences
plt.figure(figsize=(10, 6))
plt.semilogy(diff_history[:, 0], label='Diff rho')
plt.semilogy(diff_history[:, 1], label='Diff rhou')
plt.semilogy(diff_history[:, 2], label='Diff rhov')
plt.xlabel('Iteration')
plt.ylabel('Norme L2 de la différence entre 2 pas de temps successifs')
plt.title(f'Différence entre 2 pas de temps successifs\n(Mach = {Minf}, Alpha = {alpha}°)')
plt.legend()
plt.grid()
plt.show()

#%% Compute and plot Cp distribution

# Physical constants (same as in main.py)
Pinf = 101325  # Pa
Tinf = 268.3  # K
gamma = 1.4
r = 287  # J/(kg·K)

# Compute derived quantities
c = np.sqrt(gamma * r * Tinf)  # sound velocity
a = c / np.sqrt(gamma)
rhoInf = Pinf / (r * Tinf)
v_inf = Minf * c
q_inf = 0.5 * rhoInf * (v_inf**2)  # Dynamic pressure

# Load surface data
x = surf_data[:, 0]
y = surf_data[:, 1]
rho = surf_data[:, 2]

# Compute pressure and Cp
p = rho * (a**2)  # Euler isotherme: P = rho * a^2
Cp = (p - Pinf) / q_inf

# Separate intrados and extrados
extrados_mask = y >= 0
intrados_mask = y < 0

x_extrados = x[extrados_mask]
y_extrados = y[extrados_mask]
Cp_extrados = Cp[extrados_mask]

x_intrados = x[intrados_mask]
y_intrados = y[intrados_mask]
Cp_intrados = Cp[intrados_mask]

# Sort by x for proper plotting
sort_ext = np.argsort(x_extrados)
x_extrados = x_extrados[sort_ext]
Cp_extrados = Cp_extrados[sort_ext]

sort_int = np.argsort(x_intrados)
x_intrados = x_intrados[sort_int]
Cp_intrados = Cp_intrados[sort_int]

# Plot Cp distribution (inverted y-axis, aerodynamic convention)
plt.figure(figsize=(10, 6))
plt.plot(x_extrados, Cp_extrados, 'b-', linewidth=1.5, label='Extrados (Upper)')
plt.plot(x_intrados, Cp_intrados, 'r-', linewidth=1.5, label='Intrados (Lower)')
plt.gca().invert_yaxis()
plt.title(f"Coefficient de Pression $C_p$ autour du profil\n(Mach = {Minf}, Alpha = {alpha}°)")
plt.xlabel("Position x (m)")
plt.ylabel("$C_p$ (Axe inversé)")
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend()
plt.show()

# Plot pressure distribution
plt.figure(figsize=(10, 6))
plt.plot(x_intrados, p[intrados_mask][sort_int], 'r-', linewidth=2, label='Intrados (y < 0)')
plt.plot(x_extrados, p[extrados_mask][sort_ext], 'b-', linewidth=2, label='Extrados (y >= 0)')
plt.title(f"Distribution de la Pression (Intrados et Extrados)\n(Mach = {Minf}, Alpha = {alpha}°)")
plt.xlabel("Position x (m)")
plt.ylabel("Pression P (Pa)")
plt.grid(True)
plt.legend()
plt.show()
# %%
