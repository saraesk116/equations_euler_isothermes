import gmsh
import numpy as np

# gmsh elements numbering
BAR2 = 1
QUAD4 = 3
TRI3 = 2


def densify(a):
    indexes = np.unique(a, return_index=True)[1]
    _a = [a[index] for index in sorted(indexes)]
    remap = dict([(_a[i], i) for i in range(len(_a))])
    dense_a = [remap[item] for item in a]
    return dense_a, remap


def read_elements(elts, glo2loc_node_indices):
    # Unpack
    eltTypes = elts[0]
    eltTags = elts[1]
    eltNodes = elts[2]

    # 'Fix' element types but duplicating infos
    types = []
    for (itype, eltType) in enumerate(eltTypes):
        for _ in range(len(eltTags[itype])):
            types.append(eltType)

    eltTags = np.hstack(eltTags)
    eltNodes = np.hstack(eltNodes)

    # Dense numbering
    _, glo2loc_elt_indices = densify(eltTags)

    # Build element -> node connectivity
    conn = []
    inode = 0
    for icell in range(len(eltTags)):
        eltType = types[icell]

        nnodes = nnodes_of_type(eltType)
        nodes = eltNodes[inode : inode + nnodes]

        inode += nnodes

        # Convert glo -> loc
        nodes = [glo2loc_node_indices[node] for node in nodes]
        conn.append(nodes)

    return conn, types, glo2loc_elt_indices


def read_msh(filepath):
    """
    Read a 2D .msh file

    Returns
    -------
    xyz
        Array of size (:,3) containing all the nodes coordinates
    c2n
        Cell to nodes connectivity. Array of array of nodes id
    cellTypes
        Array of cell type for each cell, with GMSH convention
    ... to be completed

    """
    gmsh.initialize()
    gmsh.open(filepath)

    # Spatial dimension of the mesh
    dim = gmsh.model.getDimension()

    # Alias
    mesh = gmsh.model.mesh

    # Read nodes
    ids, xyz, _ = mesh.getNodes()
    xyz = np.reshape(xyz, (-1, 3))
    _, glo2loc_node_indices = densify(ids)

    # Read faces on tagged entities
    f2n_tagged, facetypes, glo2loc_face_indices = read_elements(
        mesh.getElements(dim - 1), glo2loc_node_indices
    )

    # Read cells
    c2n, celltypes, _ = read_elements(mesh.getElements(dim), glo2loc_node_indices)

    # Read boundary conditions (faces)
    # Warning, `bc2f` is `bc name -> tagged face`
    bc_dimtags = gmsh.model.getPhysicalGroups(dim - 1)
    bnd2f_tagged = {}
    for bnd_dimtag in bc_dimtags:
        bnddim = bnd_dimtag[0]
        bndtag = bnd_dimtag[1]
        bndname = gmsh.model.getPhysicalName(bnddim, bndtag)
        entities = gmsh.model.getEntitiesForPhysicalGroup(bnddim, bndtag)
        bndfaces = []
        for entity in entities:
            elts = mesh.getElements(bnddim, entity)
            for _tags in elts[1]:
                bndfaces.append(_tags)
        bndfaces = np.hstack(bndfaces)

        # Apply glo -> loc
        bndfaces = [glo2loc_face_indices[node] for node in bndfaces]

        bnd2f_tagged[bndname] = bndfaces

    gmsh.finalize()

    return xyz, c2n, celltypes, bnd2f_tagged, f2n_tagged


def nnodes_of_type(eltType):
    if eltType == QUAD4:
        return 4
    elif eltType == TRI3:
        return 3
    elif eltType == BAR2:
        return 2
    else:
        raise ValueError()
