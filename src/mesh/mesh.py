import numpy as np
import gmsh_reader
import sys


class Mesh:
    """
    This class represent a mesh cells composed of nodes. For now, only works for 2D mesh.
    """

    def __init__(self, coords, c2n, c2f, f2n, f2c, bnd2f, spaDim=3):
        """
        Mesh constructor

        Parameters
        ----------
        coords : ndarray
            Node coordinates. Array of size (nnodes, nspa)
        c2n : list / ndarray
            Cell to nodes connectivity : each element of this list is a list/array of the node indices
            forming this cell
        c2f : list / ndarray
            Cell to faces connectivity
        f2n : list / ndarray
            Face to nodes connectivity
        f2c : ndarray
            Face to cells connectivity. Array of size (nfaces, 2). If the face is a boundary face,
             the cell index for the "second" neighor cell is greater than the number of cells in the mesh.
        bnd2f : dict
            Boundary name to faces forming this boundary
        """
        self.coords = coords[
            :, :spaDim
        ]  # Node coordinates : array of size (nnodes, spaDim)
        self.c2n = c2n  # Cell -> node connectivity : array of arrays
        self.c2f = c2f  # cell -> face (array of arrays)
        self.f2n = f2n  # face -> node (array of size (nfaces, 2))
        self.f2c = f2c  # face -> cell (array of size (nfaces, 2))
        self.bnd2f = bnd2f  # boundary name -> faces (dict)

        # `cache` is a dictionnary intended to store information about the mesh
        # It could serve as a "cache". For instance, you may want to store the area
        # of each cell instead of recomputing it many times:
        # c2area = compute_area(...)
        # mesh.cache["c2area" : c2area]
        self.cache = {}

    def nnodes(self):
        """
        Return the number of nodes
        """
        return self.coords.shape[0]

    def ncells(self):
        """
        Return the number of cells
        """
        return len(self.c2n)

    def nfaces(self):
        """
        Return the number of faces
        """
        return self.f2n.shape[0]

    def getNSpatialDimensions(self):
        return self.coords.shape[1]

    def face_area(self, kface):
        """
        Compute area of a face. In 2D, it means the length of an edge
        """
        if "face_areas" in self.cache:
            return self.cache["face_areas"][kface]

        # Get face nodes coordinates
        n1 = self.coords[self.f2n[kface, 0], :]
        n2 = self.coords[self.f2n[kface, 1], :]

        return np.linalg.norm(n2 - n1)

    def cache_face_areas(self):
        """
        Compute all face area and cache results
        """
        areas = np.zeros(self.nfaces())
        for kface in range(self.nfaces()):
            areas[kface] = self.face_area(kface)

        self.cache["face_areas"] = areas

    def cell_volume(self, icell):
        """
        Compute volume of a cell. In 2D, it means the surface of a quad or a tri
        """
        if "cell_volumes" in self.cache:
            return self.cache["cell_volumes"][icell]

        # Get three consecutive nodes -> here we assume that nodes are correctly ordered
        n1 = self.coords[self.c2n[icell][0], :]
        n2 = self.coords[self.c2n[icell][1], :]
        n3 = self.coords[self.c2n[icell][2], :]

        # Compute cross product
        volume = abs(np.cross(n3 - n1, n2 - n1))

        # Return result depending of element type
        nnodes = len(self.c2n[icell])
        if nnodes == 3:
            return volume / 2
        elif nnodes == 4:
            return volume
        else:
            print("Area not implemented for element with {:d} nodes".format(nnodes))
            sys.exit()

    def cell_volumes(self):
        """
        Compute volume of all cells
        """
        if "cell_volumes" in self.cache:
            return self.cache["cell_volumes"]

        volumes = np.zeros(self.ncells())
        for icell in range(self.ncells()):
            volumes[icell] = self.cell_volume(icell)
        return volumes

    def cache_cell_volumes(self):
        """
        Compute all cell volumes and store in cache
        """
        self.cache["cell_volumes"] = self.cell_volumes()

    def face_center(self, iface):
        """
        Return the center of a face (i.e an edge in 2D)
        """
        inodes = self.f2n[iface]
        return np.mean(self.coords[inodes, :], axis=0)

    def cell_center(self, icell):
        """
        Return the center of cell `icell`
        """
        inodes = self.c2n[icell]
        return np.mean(self.coords[inodes, :], axis=0)

    def cell_centers(self):
        """
        Rertun an array of all cell centers of this mesh
        """
        centers = np.zeros((self.ncells(), self.getNSpatialDimensions()))
        for icell in range(self.ncells()):
            centers[icell, :] = self.cell_center(icell)
        return centers

    def face_normal(self, kface):
        """
        Return the face normal of face `iface`.
        Handle the case where `iface` is a limit face.

        Orientation is given by mesh.f2c : i -> j
        """
        if "face_normals" in self.cache:
            return self.cache["face_normals"][kface, :]

        # Get first neighbor cell
        i = self.f2c[kface, 0]

        # Get face nodes coordinates
        n1 = self.coords[self.f2n[kface, 0], :]
        n2 = self.coords[self.f2n[kface, 1], :]

        # Direction vector
        d = n2 - n1

        # Normal vector (non oriented)
        n = np.array([-d[1], d[0]])
        n = n / np.linalg.norm(n)

        # Get cell center of `i` and face center of `kface`
        ci = self.cell_center(i)
        ck = self.face_center(kface)

        # Reverse normal if not oriented as the line connecting ci to ck
        if np.dot(n, ck - ci) < 0:
            n *= -1

        return n

    def cache_face_normals(self):
        """
        Compute all face normals and store them in cache
        """
        n = np.zeros((self.nfaces(), self.getNSpatialDimensions()))
        for kface in range(self.nfaces()):
            n[kface, :] = self.face_normal(kface)

        self.cache["face_normals"] = n

    def flatten_connectivity(self):
        """
        Flatten the connectivity and produce an array of offset
        """
        # Count number of nodes in conn
        n = 0
        for inodes in self.c2n:
            n += len(inodes)

        # Allocate
        conn = np.zeros(n, dtype=int)
        offset = np.zeros(self.ncells(), dtype=int)

        # Fill !
        for (ielt, inodes) in enumerate(self.c2n):
            nnodes = len(inodes)
            if ielt > 0:
                offset[ielt] = offset[ielt-1] + nnodes
            else:
                offset[ielt] = nnodes

            conn[offset[ielt] - nnodes:offset[ielt]] = inodes

        return conn, offset


def mesh_from_msh(filepath, spaDim=3):
    """
    Build a `Mesh` reading a gmsh mesh file.
    """
    xyz, c2n, celltypes, bnd2f_tagged, f2n_tagged = gmsh_reader.read_msh(filepath)
    c2f, f2n, f2c = build_connectivities(c2n)
    bnd2f = build_bnd2faces(f2n, bnd2f_tagged, f2n_tagged)
    return Mesh(xyz, c2n, c2f, f2n, f2c, bnd2f, spaDim)


def build_connectivities(c2n):
    """
    only for 2D
    """
    # First we create the list of all the faces, excluding duplicates
    # In the same time, we build the cell to face conn
    f2c = -1 * np.ones((4 * len(c2n), 2), dtype=int)  # over-dimensionned
    f2n = -1 * np.ones((4 * len(c2n), 2), dtype=int)  # over-dimensionned
    c2f = []
    nfaces = 0
    for (icell, nodes) in enumerate(c2n):
        # print("cell {:d}".format(icell))
        _c2f = []  # local to cell
        for n1, n2 in zip(nodes, np.roll(nodes, -1)):
            pair = [n1, n2]
            iface = index_of_pair_in_array(f2n, pair)

            # If face does not exist yet
            if iface < 0:
                # print("  Face with [{:d},{:d}] does not exist".format(n1,n2))
                iface = nfaces
                f2n[iface, :] = pair
                nfaces += 1

            # Append the face to the list of faces of the current cell
            _c2f.append(iface)

            # Append the cell as a neighbor cell of the current face
            if f2c[iface][0] < 0:
                f2c[iface][0] = icell
            else:
                f2c[iface][1] = icell

        c2f.append(_c2f)

    # Resize
    f2n = f2n[:nfaces, :]
    f2c = f2c[:nfaces, :]

    # Put `ncells + 1` instead of `-1` in f2c[,1] for boundary faces to lead to a wrong index error
    # if it is used
    f2c[np.where(f2c[:,1] == -1)[0],1] = len(c2n) + 1

    return c2f, f2n, f2c


def build_bnd2faces(f2n, bnd2f_tagged, f2n_tagged):
    """
    Build the 'boundary name' -> 'face indices' connectivity. Returns a dict.
    """
    bnd2f = {}
    # bc2f = dict([(bcname, []) for bcname in bc2f_tagged.keys()])

    for (bndname, tagged_faces) in bnd2f_tagged.items():
        faces = []
        for tagged_face in tagged_faces:
            faces.append(index_of_pair_in_array(f2n, f2n_tagged[tagged_face]))
        bnd2f[bndname] = faces

    return bnd2f


def pair_in_array(array, pair):
    """
    Test if the pair (unordered) is in the array
    """
    if len(np.where((array[:, 0] == pair[0]) & (array[:, 1] == pair[1]))[0]) > 0:
        return True
    elif len(np.where((array[:, 0] == pair[1]) & (array[:, 1] == pair[0]))[0]) > 0:
        return True
    else:
        return False


def index_of_pair_in_array(array, pair):
    """
    Find the index of a pair (unordered) in the array
    """
    ind = np.where((array[:, 0] == pair[0]) & (array[:, 1] == pair[1]))[0]
    if len(ind) > 0:
        return ind[0]

    ind = np.where((array[:, 0] == pair[1]) & (array[:, 1] == pair[0]))[0]
    if len(ind) > 0:
        return ind[0]

    return -1
