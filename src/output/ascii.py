import numpy as np


def volume2ascii(filename, mesh, q):
    """ Save solution to an ascii file """
    centers = mesh.cell_centers()
    np.savetxt(filename, np.column_stack((centers[:, 0], centers[:, 1], q[:, 0], q[:, 1], q[:, 2])))


def surf2ascii(filename, mesh, surfnames, q, header="", sep=",", comments=""):
    """
    Save a surface solution to an csv ascii file
    Default args are for Paraview
    """
    s = q.shape
    if len(s) > 1:
        nvars = s[1]
    else:
        nvars = 1

    nspa = mesh.getNSpatialDimensions()

    for name in surfnames:
        kfaces = mesh.bnd2f[name]

        data = np.zeros((len(kfaces), nspa + nvars))  # coords and data on same row
        for (i, kface) in enumerate(kfaces):
            # Coords
            data[i, 0:nspa] = mesh.face_center(kface)

            # Closest point "interpolation"
            icell = mesh.f2c[kface, 0]
            if nvars == 1:
                data[i, nspa] = q[icell]
            else:
                data[i, nspa:] = q[icell, :]

    np.savetxt(filename, data, header=header, delimiter=sep, comments=comments)
