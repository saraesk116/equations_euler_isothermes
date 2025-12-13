import sys


def write2vtk(filename, coords, conn, offset, types, variables):
    """
    Write data (PointData and/or CellData) associated to an unstructured grid to a .vtu file.

    Parameters
    ----------
    filename : str
    coords : ndarray
        Node coordinates : array of size (nnodes, spa)
    conn : ndarray
        Flatten connectivity. Ex: [3, 8, 11,    4, 5, 8, 1,    7, 6, 4]
    offset : ndarray
        Vector with index of last node for each element in conn. Size is (ncells)
        Ex (coherent with prev. conn) [3, 7, 10]
    types : ndarray
        Vector of element types (vtk convention). Size is (ncells)
    variables : list
        List of list. Each 'list' item is : numpy array, data location, varname
        Ex : `variables = ((rho[:], "PointData", "rho"), (velocity[:,:], "CellData", "u"))
    """
    #
    nnodes, nspa = coords.shape
    ncells = len(offset)

    # Open file
    f = open(filename, "w")

    # Write header
    write(f, '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="BigEndian">', 0)
    write(f, '<UnstructuredGrid>', 1)
    write(f, '<Piece NumberOfPoints="{:d}" NumberOfCells="{:d}">'.format(nnodes, ncells), 2)

    # Mesh nodes
    write(f, '<Points>', 3)
    write(f, '<DataArray type="Float32" NumberOfComponents="3" Format="ascii">', 4)  # mandatory to set 3 components
    for inode in range(nnodes):
        if nspa == 1:
            write(f, '{:.6e} {:.6e} {:.6e}'.format(coords[inode, 0], 0., 0.), 5)
        elif nspa == 2:
            write(f, '{:.6e} {:.6e} {:.6e}'.format(coords[inode, 0], coords[inode, 1], 0.), 5)
        else:
            write(f, '{:.6e} {:.6e} {:.6e}'.format(coords[inode, 0], coords[inode, 1], 0.), 5)
    write(f, '</DataArray>', 3)
    write(f, '</Points>', 2)

    # Mesh cells
    write(f, '<Cells>', 3)
    write(f, '<DataArray type="Int32" Name="connectivity" Format="ascii">', 4)
    write(f, ' '.join(map(str, conn)) + '', 5)
    write(f, '</DataArray>', 4)
    write(f, '<DataArray type = "Int32" Name = "offsets" Format = "ascii" >', 4)
    write(f, ' '.join(map(str, offset)) + '', 5)
    write(f, '</DataArray>', 4)
    write(f, '<DataArray type = "Int32" Name = "types" Format = "ascii" >', 4)
    write(f, ' '.join(map(str, types)) + '', 5)
    write(f, '</DataArray>', 4)
    write(f, '</Cells>', 3)

    # Point data
    write(f, '<PointData Scalars="scalars">', 3)
    for variable in variables:
        values, location, name = variable
        if location == "PointData" and len(values.shape) == 1:
            write(f, '<DataArray type="Float32" Name="{:s}" Format="ascii">'.format(name), 4)
            write(f, ' '.join(map(str, values)), 5)
            write(f, '</DataArray>', 4)
        elif location == "PointData" and len(values.shape) > 1:
            write(f, '<DataArray type="Float32" Name="{:s}" NumberOfComponents="{:d}" Format="ascii">'.format(
                name, values.shape[1]), 4)
            for inode in range(nnodes):
                write(f, ' '.join(map(str, values[inode, :])), 5)
            write(f, '</DataArray>', 4)
    write(f, '</PointData>', 3)

    # Cell data
    write(f, '<CellData Scalars="scalars">', 3)
    for variable in variables:
        values, location, name = variable
        if location == "CellData" and len(values.shape) == 1:
            write(f, '<DataArray type="Float32" Name="{:s}" Format="ascii">'.format(name), 4)
            write(f, ' '.join(map(str, values)), 5)
            write(f, '</DataArray>', 4)
        elif location == "CellData" and len(values.shape) > 1:
            write(f, '<DataArray type="Float32" Name="{:s}" NumberOfComponents="{:d}" Format="ascii">'.format(
                name, values.shape[1]), 4)
            for inode in range(ncells):
                write(f, ' '.join(map(str, values[inode, :])), 5)
            write(f, '</DataArray>', 4)
    write(f, '</CellData>', 3)

    # End
    write(f, '</Piece>', 2)
    write(f, '</UnstructuredGrid>', 1)
    write(f, '</VTKFile>', 0)

    # Close file
    f.close()


def write(f, text, indent):
    prefix = '  ' * indent
    f.write(prefix + text + "\n")


def nnodes2vtkType(nnodes):
    """
    Only for 2D
    """
    if nnodes == 3:
        return 5
    elif nnodes == 4:
        return 9
    else:
        print("Unknown vtk type for element with {:d} nodes".format(nnodes))
        sys.exit(0)
