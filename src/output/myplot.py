# Code inspired from https://stackoverflow.com/a/59971611/2887058

import matplotlib.pyplot as plt
import matplotlib.tri as tri
from alert import *
import numpy as np
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


class MyPlot:
    def __init__(self, mesh, varnames, limvals=None, figsize=(12, 8)):
        self.mesh = mesh
        self.varnames = varnames
        self.limvals = limvals
        self.ite = 1

        # Triangulate mesh
        elts_tri = triangulate_mesh(mesh)
        self.triangulation = tri.Triangulation(mesh.coords[:, 0], mesh.coords[:, 1], elts_tri)

        # Allocate array for values on nodes
        self.qnodes = np.zeros((mesh.nnodes(), len(varnames)))

        # Compute cell volumes and nodes "volume" (for interpolation)
        self.cell_volumes = mesh.cell_volumes()
        self.node_volumes = np.zeros(mesh.nnodes())
        for (icell, inodes) in enumerate(mesh.c2n):
            vol = self.cell_volumes[icell]
            for inode in inodes:
                self.node_volumes[inode] += vol

        # Create figure and axe dict
        self.fig, axs = plt.subplots(2, 2, figsize=figsize)
        self.axs = [item for sublist in axs for item in sublist]  # flatten axes
        self.cax = [make_axes_locatable(ax).append_axes("right", size="5%", pad="2%") for ax in self.axs]

        # Tweak for contours
        self.contours = []

        # Record min/max values
        self.minmax = np.zeros((len(varnames), 2))
        for ivar in range(len(varnames)):
            self.minmax[ivar, 0] = 1e9
            self.minmax[ivar, 1] = -1e9

    def show(self):
        plt.show()

    def clear_axis(self):
        for ax in self.axs:
            ax.clear()
        for cax in self.cax:
            cax.clear()

        for contour in self.contours:
            for item in contour.collections:
                item.remove()
        self.contours = []

    def update_solution(self, qcell, autorange=False):
        """
        For now : we plot the mesh each time
        """
        self.update_nodal_values(qcell)

        self.clear_axis()

        for (ivar, varname) in enumerate(self.varnames):
            self.axs[ivar].set_title(varname)
            plot_mesh(self.axs[ivar], self.mesh)

            self.add_solution_to_plot(ivar, autorange)

    def update_and_save(self, folder, qcell):
        """
        Update the solution and save fig to file
        """
        self.update_solution(qcell)
        self.fig.savefig(folder + "/fig_{:06d}.png".format(self.ite))
        self.ite += 1

    def update_nodal_values(self, qcell):
        """
        Compute nodal values from cell-center values
        """
        # Reset qnodes
        self.qnodes[:, :] = 0

        # Loop over cells
        for (icell, inodes) in enumerate(self.mesh.c2n):
            # Loop over cell nodes
            for inode in inodes:
                self.qnodes[inode, :] += self.cell_volumes[icell] * qcell[icell, :]

        # Divide by volume
        for inode in range(self.mesh.nnodes()):
            self.qnodes[inode, :] = self.qnodes[inode, :] / self.node_volumes[inode]

        # Record min/max
        for ivar in range(self.qnodes.shape[1]):
            self.minmax[ivar, 0] = min(self.minmax[ivar, 0], np.min(self.qnodes[:, ivar]))
            self.minmax[ivar, 1] = max(self.minmax[ivar, 1], np.max(self.qnodes[:, ivar]))

    def add_solution_to_plot(self, ivar, autorange=False):
        # Create contour
        if self.limvals and (not autorange):
            vmin = self.limvals[ivar][0]
            vmax = self.limvals[ivar][1]
            contour = self.axs[ivar].tricontourf(
                self.triangulation, self.qnodes[:, ivar], np.linspace(vmin, vmax, 10))
        else:
            contour = self.axs[ivar].tricontourf(
                self.triangulation, self.qnodes[:, ivar], extend="both")

        self.contours.append(contour)

        # Create colorbar
        cb = self.fig.colorbar(contour, cax=self.cax[ivar])

        self.axs[ivar].axis('equal')


def plot_mesh(ax, mesh):
    """
    Plot the mesh (edges, in black)
    """
    for inodes in mesh.c2n:
        x = mesh.coords[inodes, 0]
        y = mesh.coords[inodes, 1]
        ax.fill(x, y, edgecolor='black', fill=False)


def triangulate_mesh(mesh):
    """
    Assumtion : only tri and quads in the mesh
    """
    tris = []

    for inodes in mesh.c2n:
        if len(inodes) == 3:
            tris.append(inodes)
        elif len(inodes) == 4:
            tris.append([inodes[0], inodes[1], inodes[2]])
            tris.append([inodes[2], inodes[3], inodes[0]])
        else:
            error("Error in `triangulate_mesh` : only tri and quad supported")

    return tris


def show_figures():
    plt.show()


def gen_fig_for_mesh_and_solution(mesh, q, qnodes):
    compute_nodal_values(mesh, q, qnodes)

    fig_rho, ax_rho, triangulation = create_new_fig(mesh, "rho")
    add_solution_to_plot(ax_rho, triangulation, qnodes[:, 0])

    fig_rhou, ax_rhou, triangulation = create_new_fig(mesh, "rhou")
    add_solution_to_plot(ax_rhou, triangulation, qnodes[:, 1])

    fig_rhov, ax_rhov, triangulation = create_new_fig(mesh, "rhov")
    add_solution_to_plot(ax_rhov, triangulation, qnodes[:, 2])

    return fig_rho, fig_rhou, fig_rhov


def plot_and_display(mesh, q, qnodes):
    """
    Plot solution and display it
    """
    gen_fig_for_mesh_and_solution(mesh, q, qnodes)
    show_figures()


def plot_and_save(folder, ite, mesh, q, qnodes):
    """
    Plot solution and save result to a file instead of displaying it
    """
    fig_rho, fig_rhou, fig_rhov = gen_fig_for_mesh_and_solution(mesh, q, qnodes)

    fig_rho.savefig(folder + "/rho_{:06d}.png".format(ite))
    fig_rhou.savefig(folder + "/rhou_{:06d}.png".format(ite))
    fig_rhov.savefig(folder + "/rhov_{:06d}.png".format(ite))

    # plt.close("all")
    fig_rho.clear()
    fig_rhou.clear()
    fig_rhov.clear()
