#
#
# H----------G------------------F
# |          .                  |
# |          .                  |
# |          .                  |
# A----------B..................E
#            |                  |
#            |                  |
#            |                  |
#            C------------------D
#
#
#

import gmsh
import os

# Mesh settings
filename = "mesh_step.msh"
lc = 0.1  # For tri mesh
nx_upstream = 10
ny_upstream = 8
nx_downstream = 10
ny_downstream = 6
l_AH = 1
l_DF = 2
l_AB = 1
l_CD = 1
structured = True  # generate structured mesh?
recombine = True  # recombine mesh (struct/unstr)

# Play with params
b = l_DF - l_AH

# Init gmsh
gmsh.initialize()

# Alias
geo = gmsh.model.geo

# Points
A = geo.addPoint(0, 0, 0, lc)
B = geo.addPoint(l_AB, 0, 0, lc)
C = geo.addPoint(l_AB, -b, 0, lc)
D = geo.addPoint(l_AB+l_CD, -b, 0, lc)
E = geo.addPoint(l_AB+l_CD, 0, 0, lc)
F = geo.addPoint(l_AB+l_CD, l_AH, 0, lc)
G = geo.addPoint(l_AB, l_AH, 0, lc)
H = geo.addPoint(0, l_AH, 0, lc)

# Lines
AB = geo.addLine(A, B)
BC = geo.addLine(B, C)
CD = geo.addLine(C, D)
DE = geo.addLine(D, E)
EF = geo.addLine(E, F)
FG = geo.addLine(F, G)
GH = geo.addLine(G, H)
HA = geo.addLine(H, A)

BG = geo.addLine(B, G)
BE = geo.addLine(B, E)

# Surfaces (for transfinite and unstructured)
if structured:
    ABGH = geo.addPlaneSurface([geo.addCurveLoop([AB, BG, GH, HA])])
    BEFG = geo.addPlaneSurface([geo.addCurveLoop([BE, EF, FG, -BG])])
    CDEB = geo.addPlaneSurface([geo.addCurveLoop([CD, DE, -BE, BC])])
else:
    surf = geo.addPlaneSurface([geo.addCurveLoop([AB, BC, CD, DE, EF, FG, GH, HA])])

# Sync
geo.synchronize()

# Mesh
mesh = gmsh.model.mesh

if structured:
    # Upstream, x direction
    n = nx_upstream
    mesh.setTransfiniteCurve(AB, n)
    mesh.setTransfiniteCurve(GH, n)

    # Upstream, y direction
    n = ny_upstream
    mesh.setTransfiniteCurve(HA, n)
    mesh.setTransfiniteCurve(BG, n)
    mesh.setTransfiniteCurve(EF, n)

    # Downstream, x direction
    n = nx_downstream
    mesh.setTransfiniteCurve(CD, n)
    mesh.setTransfiniteCurve(BE, n)
    mesh.setTransfiniteCurve(FG, n)

    # Step, y direction
    n = ny_downstream
    mesh.setTransfiniteCurve(BC, n)
    mesh.setTransfiniteCurve(DE, n)

    # Surfaces
    for s in [ABGH, BEFG, CDEB]:
        mesh.setTransfiniteSurface(s)
        if recombine:
            mesh.setRecombine(2, s)
else:
    if recombine:
        mesh.setRecombine(2, surf)

mesh.generate(2)

# Boundary conditions
bnd_in = gmsh.model.addPhysicalGroup(1, [HA])
gmsh.model.setPhysicalName(1, bnd_in, "INLET")

bnd_out = gmsh.model.addPhysicalGroup(1, [DE, EF])
gmsh.model.setPhysicalName(1, bnd_out, "OUTLET")

bnd_wall = gmsh.model.addPhysicalGroup(1, [AB, BC, CD, FG, GH])
gmsh.model.setPhysicalName(1, bnd_wall, "WALL")

if structured:
    domain = gmsh.model.addPhysicalGroup(2, [ABGH, BEFG, CDEB])
else:
    domain = gmsh.model.addPhysicalGroup(2, [surf])
gmsh.model.setPhysicalName(2, domain, "DOMAIN")

# Write file
gmsh.write(os.path.dirname(__file__) + "/" + filename)

# Then end !
gmsh.finalize()
