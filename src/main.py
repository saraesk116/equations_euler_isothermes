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

#%%------------------------------------------------------ Settings
nite = 2000 # Number of time iterations
CFL = 0.2  # CFL number
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

#%%------------------------------------------------------ Prepare to run
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
#%%----------------------------------------------------- Run simulation
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

# Save results with timestamp and parameters
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
local_str = "local_True" if localTimeStep else "local_False"
results_dir = curr_dir + f"/../results/{timestamp}_CFL_{CFL}_nite_{nite}_{local_str}_alpha_{alpha}_Minf_{Minf}"
os.makedirs(results_dir, exist_ok=True)

# Save residuals and differences in npz format
np.savez(
    results_dir + "/convergence_data.npz",
    residus_history=np.array(residus_history),
    diff_history=np.array(diff_history),
    nite=nite,
    CFL=CFL,
    localTimeStep=localTimeStep,
    Minf=Minf,
    alpha=alpha,
    elapsed_time_seconds=elapsed_time.total_seconds()
)

# Save simulation setup to text file
with open(results_dir + "/simulation_setup.txt", "w") as f:
    f.write("=" * 50 + "\n")
    f.write("SIMULATION SETUP\n")
    f.write("=" * 50 + "\n")
    f.write(f"Timestamp: {timestamp}\n")
    f.write(f"Elapsed time: {elapsed_time}\n")
    f.write("\n--- Physical Parameters ---\n")
    f.write(f"Pinf = {Pinf:.3e} Pa\n")
    f.write(f"Tinf = {Tinf:.3e} K\n")
    f.write(f"Minf = {Minf:.3e}\n")
    f.write(f"alpha = {alpha} deg\n")
    f.write(f"rhoInf = {rhoInf:.3e} kg/m³\n")
    f.write(f"rhouInf = {rhouInf:.3e} kg/(m²·s)\n")
    f.write(f"rhovInf = {rhovInf:.3e} kg/(m²·s)\n")
    f.write(f"gamma = {gamma:.3e}\n")
    f.write(f"r = {r:.3e} J/(kg·K)\n")
    f.write(f"a = {a:.3e}\n")
    f.write(f"c = {c:.3e} m/s\n")
    f.write("\n--- Numerical Parameters ---\n")
    f.write(f"nite = {nite}\n")
    f.write(f"CFL = {CFL:.2e}\n")
    f.write(f"Local time step = {localTimeStep}\n")
    f.write(f"Spatial dimension = {spaDim}\n")
    f.write("\n--- Mesh Information ---\n")
    f.write(f"Mesh file: {mesh_path}\n")
    f.write(f"Number of cells = {mesh.ncells()}\n")
    f.write(f"Number of nodes = {mesh.nnodes()}\n")
    f.write(f"Number of faces = {mesh.nfaces()}\n")
    f.write("=" * 50 + "\n")

# Output solution to file in results directory
surf2ascii(results_dir + "/surf.csv", mesh, ["WALL"], q[:, 0], header="x,y,rho")

# Export to vtk
conn, offset = mesh.flatten_connectivity()
types = [nnodes2vtkType(len(inodes)) for inodes in mesh.c2n]
variables = (
    (q[:, 0], "CellData", "rho"),
    (q[:, 1:], "CellData", "rhoU"),
)
write2vtk(curr_dir + "/../results/result.vtu", mesh.coords, conn, offset, types, variables)


print(f"Results saved in: {results_dir}")

#%%------------------------------------------------------ Post-process
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

# Keep backward compatibility - also save to root directory
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

np.savetxt(curr_dir + "/../residus_history.csv", residus_history, delimiter=",", header="residu_rho,residu_rhou,residu_rhov", comments='')
np.savetxt(curr_dir + "/../diff_history.csv", diff_history, delimiter=",", header="diff_rho,diff_rhou,diff_rhov", comments='')


#%%Compute lift and drag

nite = 3000  # Number of time iterations
angles_incidence = np.linspace(-5, 15, 5)  # angles d'incidence en degrés
Cl_values = []
Cd_values = []

# On suppose une corde unitaire si non spécifiée
chord = 1.0 

# Dictionnaire pour stocker l'historique des résidus par angle
residuals_by_alpha = {}

for alpha in angles_incidence:

    # -- SETUP SIMULATION WITH NEW ALPHA --
    radian_alpha = alpha * pi / 180
    rhouInf = rhoInf * Minf * c * cos(radian_alpha)
    rhovInf = rhoInf * Minf * c * sin(radian_alpha)

    # Re-initialize solution
    q[:, 0] = rhoInf
    q[:, 1] = rhouInf
    q[:, 2] = rhovInf

    # Update params
    params["rhouInf"] = rhouInf
    params["rhovInf"] = rhovInf

    current_residuals = [] # To store residuals for current alpha

    # -- RUN SIMULATION --
    for t in tqdm(range(nite), desc=f"Angle d'incidence {alpha}°"):
        q_old = q.copy()
        solve_one_time_step(mesh, q, flux, dt, params)

        # Compute residuals 
        diff = q - q_old
        # small epsilon to avoid division by zero
        norm_q = np.linalg.norm(q_old[:, 0])
        if norm_q < 1e-12: norm_q = 1.0
            
        residu_rho = np.linalg.norm(diff[:, 0]) / norm_q
        current_residuals.append(residu_rho)

    # Store residuals for this angle
    residuals_by_alpha[alpha] = current_residuals
    
    # -- POST-PROCESS TO COMPUTE Cl AND Cd --
    Fx = 0.0
    Fy = 0.0

    # Pression dynamique infinie pour normalisation (0.5 * rho * V^2)
    q_dyn_inf = 0.5 * rhoInf * (Minf * c)**2

    for bnd_name, face_ids in mesh.bnd2f.items():
        if bnd_name == "WALL":
            for face_id in face_ids:
                # CORRECTION : Accès direct au tableau f2c avec l'indice de la face
                # f2c est (nfaces, 2). Pour une frontière, col 0 = cellule interne.
                cell_id = mesh.f2c[face_id, 0]
                rho = q[cell_id, 0]
                
                # Calcul Pression (Euler Isotherme : P = rho * a^2)
                P = rho * params["a"]**2
                
                # Coefficient de pression
                Cp = (P - Pinf) / q_dyn_inf
                
                # Récupération de la géométrie via les méthodes de la classe Mesh
                normal = mesh.face_normal(face_id)
                nx = normal[0]
                ny = normal[1]
                L = mesh.face_area(face_id)
                
                Fx += Cp * nx * L
                Fy += Cp * ny * L
    
    # Coefficients dans le repère du corps (Cx, Cy)
    Cx = Fx / chord
    Cy = Fy / chord
    
    # Projection dans le repère vent (Lift, Drag)
    cl_actuel = -Cx * sin(radian_alpha) + Cy * cos(radian_alpha) # Portance
    cd_actuel = Cx * cos(radian_alpha) + Cy * sin(radian_alpha) # Trainée
    
    Cl_values.append(cl_actuel)
    Cd_values.append(cd_actuel)
    print(f"  -> Cl = {cl_actuel:.4f}, Cd = {cd_actuel:.4f}, Résidu final = {current_residuals[-1]:.2e}")

#%% Plot residuals vs iterations for each alpha

dir_plot = curr_dir + "/../plots_incidence/"
if not os.path.exists(dir_plot):
    os.makedirs(dir_plot)


# Save residuals history for each alpha in npz (convert numeric keys to strings)
residuals_str_keys = {f"alpha_{alpha}": res for alpha, res in residuals_by_alpha.items()}
np.savez(dir_plot + "residuals_by_alpha.npz", **residuals_str_keys)

# save Cl and Cd values
np.savez(dir_plot + "Cl_Cd_values.npz", angles_incidence=angles_incidence, Cl_values=Cl_values, Cd_values=Cd_values)

# Tracé de la convergence pour chaque angle
plt.figure(figsize=(8,6))
for alpha, res in residuals_by_alpha.items():
    plt.semilogy(res, label=f'Alpha = {alpha}°')

plt.xlabel('Itération')
plt.ylabel('Résidu (rho)')
plt.title('Convergence pour différents angles d\'incidence')
plt.legend()
plt.grid()
plt.savefig(dir_plot + "convergence_incidence.png")
# save figure in npz
plt.show()

# Plot Cl and Cd vs alpha
plt.figure(figsize=(8,6))
plt.plot(angles_incidence, Cl_values, '-o', label='Cl')
plt.xlabel('Incidence (deg)')
plt.ylabel('Coefficient de Portance (Cl)')
plt.title('Polaire de portance')
plt.grid()
plt.legend()
plt.savefig(dir_plot + "polaire_portance.png")
plt.show()

plt.figure(figsize=(8,6))
plt.plot(angles_incidence, Cd_values, '-o', label='Cd')
plt.xlabel('Incidence (deg)')
plt.ylabel('Coefficient de Trainée (Cd)')
plt.title('Polaire de trainée')
plt.grid()
plt.legend()
plt.savefig(dir_plot + "polaire_trainée.png")
plt.show()
# %%
