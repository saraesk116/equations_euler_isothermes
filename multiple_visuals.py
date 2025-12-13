# autopep8: off
#%% Imports
import os
import numpy as np
import matplotlib.pyplot as plt

#%% Load files

NAME_DIR1 = "20251213_170329_CFL_0.2_nite_2000_local_False_alpha_4_Minf_0.273"
NAME_DIR2 = "20251213_172425_CFL_0.2_nite_2000_local_True_alpha_4_Minf_0.273"

# Choose which variable to compare (diff rho, diff rhou, diff rhov)
VAR_TO_COMPARE = 'diff rhou'  # Options: 'diff rho', 'diff rhou', 'diff rhov'



PATH_DIR1 = os.path.dirname(os.path.realpath(__file__)) + f"/results/{NAME_DIR1}/"
PATH_DIR2 = os.path.dirname(os.path.realpath(__file__)) + f"/results/{NAME_DIR2}/"

# Load data
convergence_data1 = np.load(PATH_DIR1 + "convergence_data.npz")
surf_data1 = np.loadtxt(PATH_DIR1 + "surf.csv", delimiter=",", skiprows=1)

convergence_data2 = np.load(PATH_DIR2 + "convergence_data.npz")
surf_data2 = np.loadtxt(PATH_DIR2 + "surf.csv", delimiter=",", skiprows=1)

#%% Plot convergence data comparison

# Determine which variable to compare
if VAR_TO_COMPARE == 'diff rho':
    VAR_INDEX = 0
elif VAR_TO_COMPARE == 'diff rhou':
    VAR_INDEX = 1
elif VAR_TO_COMPARE == 'diff rhov':
    VAR_INDEX = 2
else:
    raise ValueError("Invalid VAR_TO_COMPARE. Choose from 'diff rho', 'diff rhou', 'diff rhov'.")

# Extract parameters from both simulations
params1 = {
    'nite': int(convergence_data1['nite']),
    'CFL': float(convergence_data1['CFL']),
    'Minf': float(convergence_data1['Minf']),
    'alpha': float(convergence_data1['alpha']),
    'localTimeStep': bool(convergence_data1['localTimeStep'])
}

params2 = {
    'nite': int(convergence_data2['nite']),
    'CFL': float(convergence_data2['CFL']),
    'Minf': float(convergence_data2['Minf']),
    'alpha': float(convergence_data2['alpha']),
    'localTimeStep': bool(convergence_data2['localTimeStep'])
}

# Find differences in parameters
diff_params = []
for key in params1.keys():
    if params1[key] != params2[key]:
        diff_params.append(f"{key}: {params1[key]} vs {params2[key]}")

# Create title and legend labels
if diff_params:
    title_suffix = "\nDifferences: " + ", ".join(diff_params)
else:
    title_suffix = "\n(Identical parameters)"

label1 = f"Sim1: {(f" ({diff_params[0].split(':')[0]}={params1[diff_params[0].split(':')[0]]})" if diff_params else "")}"
label2 = f"Sim2: {(f" ({diff_params[0].split(':')[0]}={params2[diff_params[0].split(':')[0]]})" if diff_params else "")}"

# Plot comparison - Differences
plt.figure(figsize=(10, 6))
plt.semilogy(convergence_data1['diff_history'][:, VAR_INDEX], label=label1, linewidth=2)
plt.semilogy(convergence_data2['diff_history'][:, VAR_INDEX], label=label2, linewidth=2)
plt.xlabel('Iteration')
plt.ylabel(f'Norme L2 de {VAR_TO_COMPARE}')
plt.title(f'Comparison: {VAR_TO_COMPARE}{title_suffix}')
plt.legend()
plt.grid()
plt.show()

# %%
