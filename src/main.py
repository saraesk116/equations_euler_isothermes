# Add modules to path
# autopep8: off
#%% 
import sys, os
curr_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, curr_dir + '/mesh')
sys.path.insert(1, curr_dir + '/solver')
sys.path.insert(1, curr_dir + '/util')
sys.path.insert(1, curr_dir + '/output')

from mesh import mesh_from_msh
import numpy as np
from math import pi, sqrt, cos, sin
from solver import solve_one_time_step
from myplot import MyPlot
from vtk import *
from ascii import *
import alert
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
# autopep8: on

#%%
#------------------------------------------------------ Settings
nite = 500 # Number of time iterations
CFL = 0.5  # CFL number
mesh_path = curr_dir + "/../mesh/naca0012.msh"  # Path to gmsh mesh file
nfigures = 0  # Number of figures desired
localTimeStep = False  # Local time step or global time step (=same for all cells)

# for "farfield" condition
Pinf = 101325  # Pressure (Pa)
Tinf = 268.3  # Temperature (K)

Minf = 0.273  # Mach number 1.1 
alpha = 4  # Angle of Attack (deg)

# for "inlet" condition
# rhouInf = 1  # x-direction
# rhovInf = 0  # y-direction

# for "outlet" pressure condition
# Pinf = 101325  # Pressure (Pa)

# Physical / geometrical constants
spaDim = 2  # Number of spatial dimensions
gamma = 1.4  # Specific heat ratio (usually 1.4)
r = 287  # Perfect gas constant

#------------------------------------------------------ Prepare to run
# Process parameters
c = sqrt(gamma * r * Tinf)  # sound velocity
a = c / sqrt(gamma)
rhoInf = Pinf / (r * Tinf)
rhouInf = rhoInf * Minf * c * cos(alpha*pi/180)
rhovInf = rhoInf * Minf * c * sin(alpha*pi/180)

# Pack into dict
params = {
    "CFL": CFL,
    "rhoInf": rhoInf,
    "rhouInf": rhouInf,
    "rhovInf": rhovInf,
    "a": a,
    "gamma": gamma,
    "localTimeStep": localTimeStep
}

# Read mesh
mesh = mesh_from_msh(mesh_path, spaDim)

# Allocate solution: `q` is a matrix (ncell, 3).
# Each row correspond to (rho, rhou, rhov) in a given cell
q = np.zeros((mesh.ncells(), 3))

# Initialize solution
q[:, 0] = rhoInf
q[:, 1] = rhouInf
q[:, 2] = rhovInf

# Allocations for performance
flux = np.zeros((mesh.nfaces(), q.shape[1]))  # Flux array
qnodes = np.zeros((mesh.nnodes(), q.shape[1]))  # Solution values on mesh nodes
dt = np.zeros(mesh.ncells())  # Time step in each cell (will evolve with CFL criteria)

residus_history = [] 
diff_history = []
#------------------------------------------------------ Run simulation
# Loop over time
#alert.incomplete("src/main.py:loop_over_time")
# get current time
from datetime import datetime
start_time = datetime.now()

for t in tqdm(range(nite), desc="Time iteration"): 
    q_old = q.copy()  # Save old solution (for residual computation)
    solve_one_time_step(mesh, q, flux, dt, params)
    diff = q - q_old
    residu_rho = np.linalg.norm(diff[:, 0], ord=2)/np.linalg.norm(q_old[:, 0], ord=2)
    residu_rhou = np.linalg.norm(diff[:, 1], ord=2)/np.linalg.norm(q_old[:, 1], ord=2)
    residu_rhov = np.linalg.norm(diff[:, 2], ord=2)/np.linalg.norm(q_old[:, 2], ord=2)
    residus_history.append((residu_rho, residu_rhou, residu_rhov))
    diff_rho = np.linalg.norm(diff[:, 0], ord=2)
    diff_rhou = np.linalg.norm(diff[:, 1], ord=2)
    diff_rhov = np.linalg.norm(diff[:, 2], ord=2)
    diff_history.append((diff_rho, diff_rhou, diff_rhov))

end_time = datetime.now()
elapsed_time = end_time - start_time
print("Elapsed time during the simulation: {}".format(elapsed_time))
#------------------------------------------------------ Post-process
# Recall simulation setup
print("----------------------")
print("Simulation parameters:")
print("Pinf = {:.3e}".format(Pinf))
print("Minf = {:.3e}".format(Minf))
print("rhoInf = {:.3e}".format(rhoInf))
print("rhouInf = {:.3e}".format(rhouInf))
print("rhovInf = {:.3e}".format(rhovInf))
print("nite = {:d}".format(nite))
print("CFL = {:.2e}".format(CFL))
print("gamma = {:.3e}".format(gamma))
print("a = {:.3e}".format(a))
print("Number of cells = {:d}".format(mesh.ncells()))
print("Local time step ? " + str(localTimeStep))

# Output solution to file
surf2ascii(curr_dir + "/../surf.csv", mesh, ["WALL"], q[:, 0], header="x,y,rho")


# Calcul de la pression P et du Coefficient de Pression (Cp)
#  CHARGEMENT ET PRÉPARATION DES DONNÉES 
# Lecture du fichier CSV
try:
    df = pd.read_csv('../surf.csv')
    # Nettoyage des noms de colonnes (enlève les espaces éventuels)
    df.columns = df.columns.str.strip()
except FileNotFoundError:
    print("Erreur : Le fichier 'surf.csv' est introuvable.")
    exit()

# Calcul de la pression P et du Coefficient de Pression (Cp)
# Cp = (P - P_inf) / q_inf
df['p'] = df['rho'] * (a**2) 
# Vitesse totale (magnitude)
v_inf = Minf * c 

# Dénominateur (Pression dynamique)
q_inf = 0.5 * rhoInf * (v_inf**2)

df['Cp'] = (df['p'] - Pinf) / q_inf

# SÉPARATION INTRADOS / EXTRADOS
# Hypothèse : Le profil est aligné sur l'axe X.
# Extrados (Upper) : y >= 0
# Intrados (Lower) : y < 0
# On trie par 'x' pour avoir un tracé de ligne propre.

extrados = df[df['y'] >= 0].sort_values(by='x')  
intrados = df[df['y'] < 0].sort_values(by='x')



# Export to vtk
conn, offset = mesh.flatten_connectivity()
types = [nnodes2vtkType(len(inodes)) for inodes in mesh.c2n]
variables = (
    (q[:, 0], "CellData", "rho"),
    (q[:, 1:], "CellData", "rhoU"),
)
write2vtk(curr_dir + "/../result.vtu", mesh.coords, conn, offset, types, variables)


#%%
# plot pression et Cp figures


# === FIGURE 1 : Pression Intrados et Extrados ===
plt.figure(figsize=(10, 6))
plt.plot(intrados['x'], intrados['p'], 'r-', linewidth=2, label='Intrados (y < 0)')
plt.plot(extrados['x'], extrados['p'], 'b-', linewidth=2, label='Extrados (y >= 0)')
plt.title("Distribution de la Pression (Intrados et Extrados)")
plt.xlabel("Position x")
plt.ylabel("Pression P")
plt.grid(True)
plt.legend()
plt.show()

# === FIGURE 2 : Coefficient de Pression Cp en fonction de x ===
plt.figure(figsize=(10, 6))

# 1. Tracer l'Extrados (souvent en bleu ou noir)
# C'est la face supérieure qui porte l'avion (généralement Cp négatif)
plt.plot(extrados['x'], extrados['Cp'], 'b-', linewidth=1.5, label='Extrados (Upper)')

# 2. Tracer l'Intrados (souvent en rouge)
# C'est la face inférieure (généralement Cp positif ou proche de 0)
plt.plot(intrados['x'], intrados['Cp'], 'r-', linewidth=1.5, label='Intrados (Lower)')

# 3. INVERSER L'AXE Y (Convention Aérodynamique) !!
# Le "vide" (succion) tire l'aile vers le haut, donc on met le -Cp en haut.
plt.gca().invert_yaxis()

# 4. Mise en forme
plt.title(f"Coefficient de Pression $C_p$ autour du profil\n(Mach = {Minf}, Alpha = {alpha}°)")
plt.xlabel("Position x (m)")
plt.ylabel("$C_p$ (Axe inversé)")
plt.axhline(0, color='black', linewidth=0.8, linestyle='--') # Ligne de référence Cp=0
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend()

plt.show()



# plot diff history
diff_history = np.array(diff_history)
plt.figure()
plt.semilogy(diff_history[:,0], label='Diff rho')
plt.semilogy(diff_history[:,1], label='Diff rhou')
plt.semilogy(diff_history[:,2], label='Diff rhov')
plt.xlabel('Iteration')
plt.ylabel('Norme L2 de la différence entre 2 pas de temps successifs')
plt.title('Différence entre 2 pas de temps successifs')
plt.legend()
plt.grid()
plt.show()


#plot residuals
residus_history = np.array(residus_history)
plt.figure()
plt.semilogy(residus_history[:,0], label='Residu rho')
plt.semilogy(residus_history[:,1], label='Residu rhou')
plt.semilogy(residus_history[:,2], label='Residu rhov')
plt.xlabel('Iteration')
plt.ylabel('Résidu (norme L2)')
plt.title('Convergence de la simulation CFD')
plt.legend()
plt.grid()
plt.show()


