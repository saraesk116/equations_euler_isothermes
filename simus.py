import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, cos, sin, pi

# --- 1. PARAMÈTRES DU PROBLÈME  ---

Pinf = 101325  # Pressure (Pa)
Tinf = 268.3  # Temperature (K)
Minf = 0.273 # Mach number
alpha = 4  # Angle of Attack (deg)
spaDim = 2  # Number of spatial dimensions
gamma = 1.4  # Specific heat ratio (usually 1.4)
r = 287  # Perfect gas constant
c = sqrt(gamma * r * Tinf)  # sound velocity
a = c / sqrt(gamma)
rhoInf = Pinf / (r * Tinf)
rhouInf = rhoInf * Minf * c * cos(alpha*pi/180)
rhovInf = rhoInf * Minf * c * sin(alpha*pi/180)

# --- 2. CHARGEMENT ET PRÉPARATION DES DONNÉES ---
# Lecture du fichier CSV
try:
    df = pd.read_csv('surf.csv')
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

# --- 3. SÉPARATION INTRADOS / EXTRADOS ---
# Hypothèse : Le profil est aligné sur l'axe X.
# Extrados (Upper) : y >= 0
# Intrados (Lower) : y < 0
# On trie par 'x' pour avoir un tracé de ligne propre.

extrados = df[df['y'] >= 0].sort_values(by='x')  
intrados = df[df['y'] < 0].sort_values(by='x')

# --- 4. TRACÉ DES 3 FIGURES ---

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


# Analyse de converge de la simulation CFD

