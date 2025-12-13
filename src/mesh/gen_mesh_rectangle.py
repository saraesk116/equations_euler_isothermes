#
#
# D--------------------C
# |                    |
# |                    |
# |                    |
# A--------------------B
#

import gmsh
import os

# Mesh settings
filename = "../../mesh/mesh_rectangle.msh"
lc = 0.1  # For tri mesh
lx = 3  # length in x-direction
ly = 1  # length in y-direction
nx = 4  # Number of points in the x-direction
ny = 2  # Number of points in the y-direction
structured = True  # generate structured mesh?
recombine = True  # recombine mesh (struct/unstr)

# Init gmsh
gmsh.initialize()

# Alias
geo = gmsh.model.geo

# Points
A = geo.addPoint(0, 0, 0, lc)
B = geo.addPoint(lx, 0, 0, lc)
C = geo.addPoint(lx, ly, 0, lc)
D = geo.addPoint(0, ly, 0, lc)

# Lines
AB = geo.addLine(A, B)
BC = geo.addLine(B, C)
CD = geo.addLine(C, D)
DA = geo.addLine(D, A)

# Surfaces (for transfinite and unstructured)
ABCD = geo.addPlaneSurface([geo.addCurveLoop([AB, BC, CD, DA])])

# Sync
geo.synchronize()

# Mesh
mesh = gmsh.model.mesh

if structured:
    # x direction
    mesh.setTransfiniteCurve(AB, nx)
    mesh.setTransfiniteCurve(CD, nx)

    # y direction
    mesh.setTransfiniteCurve(BC, ny)
    mesh.setTransfiniteCurve(DA, ny)

    # Surface
    mesh.setTransfiniteSurface(ABCD)


if recombine:
    mesh.setRecombine(2, ABCD)

mesh.generate(2)

# Boundary conditions
bnd_in = gmsh.model.addPhysicalGroup(1, [DA])
gmsh.model.setPhysicalName(1, bnd_in, "INLET")

bnd_out = gmsh.model.addPhysicalGroup(1, [BC])
gmsh.model.setPhysicalName(1, bnd_out, "OUTLET")

bnd_wall = gmsh.model.addPhysicalGroup(1, [AB, CD])
gmsh.model.setPhysicalName(1, bnd_wall, "WALL")

domain = gmsh.model.addPhysicalGroup(2, [ABCD])
gmsh.model.setPhysicalName(2, domain, "DOMAIN")

# Write file
gmsh.write(os.path.dirname(__file__) + "/" + filename)

# Then end !
gmsh.finalize()
