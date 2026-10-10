"""Global assembly, loads, boundary conditions and the linear solve
(DefineForceMatrix.m, DefineConstrains.m, DefineSolver.m)."""
from __future__ import annotations
import warnings
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu


def assemble(Ke, dof, ndof):
    """Global stiffness (CSC) from element matrices Ke (ne,8,8) and dof map (ne,8)."""
    rows = np.repeat(dof, 8, axis=1).ravel()
    cols = np.tile(dof, (1, 8)).ravel()
    K = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(ndof, ndof))
    return K.tocsc()


def force_vector(mesh, pressure, radius):
    """Nodal forces of the circular pressure on the surface (ring areas, divided by 2 to match
    the pi*r integration of the stiffness)."""
    nodes = mesh.nodes
    tol = 1e-8 * max(mesh.width, 1.0)
    F = np.zeros(2 * nodes.shape[0])
    active = np.flatnonzero((np.abs(nodes[:, 1]) <= tol) & (nodes[:, 0] <= radius + tol))
    rr = nodes[active, 0]
    order = np.argsort(rr); active, rr = active[order], rr[order]
    n = rr.size
    for i in range(n):
        if i == 0:
            r2 = ((rr[0] + rr[1]) / 2) ** 2
        elif i == n - 1:
            r2 = radius ** 2 - ((rr[i - 1] + rr[i]) / 2) ** 2
        else:
            r2 = ((rr[i] + rr[i + 1]) / 2) ** 2 - ((rr[i - 1] + rr[i]) / 2) ** 2
        F[2 * active[i] + 1] = -np.pi * r2 * pressure / 2
    expected = -pressure * np.pi * radius ** 2 / 2
    if abs(F.sum() - expected) > 0.01 * abs(expected):
        warnings.warn("Applied load (%.1f) differs from p*pi*a^2/2 (%.1f) by more than 1%%; "
                      "check the mesh at r = a." % (F.sum(), expected))
    return F


def fixed_dofs(mesh, inner=("fix", "free"), bottom=("fix", "fix"), outer=("fix", "free")):
    """DOF indices removed by the boundary conditions (U1 = u_r, U2 = u_z)."""
    nodes = mesh.nodes
    tol = 1e-8 * max(mesh.width, mesh.depth)
    out = []
    for mask, (u1, u2) in [(np.abs(nodes[:, 0]) <= tol, inner),
                           (np.abs(nodes[:, 1] + mesh.depth) <= tol, bottom),
                           (np.abs(nodes[:, 0] - mesh.width) <= tol, outer)]:
        idx = np.flatnonzero(mask)
        if u1 == "fix":
            out.append(2 * idx)
        if u2 == "fix":
            out.append(2 * idx + 1)
    return np.unique(np.concatenate(out)) if out else np.array([], int)


class Solver:
    """Reduced-system direct solver; the sparsity pattern is fixed, only values change."""

    def __init__(self, ndof, fixed):
        self.ndof = ndof
        self.free = np.setdiff1d(np.arange(ndof), fixed)

    def solve(self, K, F):
        Kr = K[self.free][:, self.free].tocsc()
        try:
            lu = splu(Kr, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                      options=dict(SymmetricMode=True))
            ur = lu.solve(F[self.free])
        except RuntimeError:
            from scipy.sparse.linalg import spsolve
            ur = spsolve(Kr, F[self.free])
        u = np.zeros(self.ndof)
        u[self.free] = ur
        return u
