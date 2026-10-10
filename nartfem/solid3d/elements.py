"""Hex-8 (trilinear) solid element, vectorized over elements.

Voigt order: [e_xx, e_yy, e_zz, g_xy, g_yz, g_xz] (engineering shear).  The mesh is a tensor
product of boxes, so the Jacobian of every element is diagonal and constant: J = diag(dx, dy, dz)/2.
"""
from __future__ import annotations
import numpy as np
from numpy.polynomial.legendre import leggauss

# natural coordinates of the 8 nodes (same order as mesh.py)
_XI = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)


def shape_function(xi, eta, zeta):
    """N (8,), dN (8,3) = dN/d(xi, eta, zeta)."""
    N = 0.125 * (1 + _XI[:, 0] * xi) * (1 + _XI[:, 1] * eta) * (1 + _XI[:, 2] * zeta)
    dN = 0.125 * np.column_stack([_XI[:, 0] * (1 + _XI[:, 1] * eta) * (1 + _XI[:, 2] * zeta),
                                  _XI[:, 1] * (1 + _XI[:, 0] * xi) * (1 + _XI[:, 2] * zeta),
                                  _XI[:, 2] * (1 + _XI[:, 0] * xi) * (1 + _XI[:, 1] * eta)])
    return N, dN


def b_matrix(dN, size):
    """B (ne,6,24) for box elements of sizes (ne,3) at a natural point with derivatives dN (8,3)."""
    ne = size.shape[0]
    dNdx = dN[None, :, :] * (2.0 / size)[:, None, :]            # (ne,8,3)
    B = np.zeros((ne, 6, 24))
    for k in range(8):
        c = 3 * k
        B[:, 0, c] = dNdx[:, k, 0]
        B[:, 1, c + 1] = dNdx[:, k, 1]
        B[:, 2, c + 2] = dNdx[:, k, 2]
        B[:, 3, c] = dNdx[:, k, 1]; B[:, 3, c + 1] = dNdx[:, k, 0]
        B[:, 4, c + 1] = dNdx[:, k, 2]; B[:, 4, c + 2] = dNdx[:, k, 1]
        B[:, 5, c] = dNdx[:, k, 2]; B[:, 5, c + 2] = dNdx[:, k, 0]
    return B


def d_matrix(E, nu):
    """Isotropic 3D rigidity matrices (ne,6,6)."""
    E = np.asarray(E, float); nu = np.asarray(nu, float)
    lam = E * nu / ((1 + nu) * (1 - 2 * nu)); mu = E / (2 * (1 + nu))
    D = np.zeros((E.size, 6, 6))
    for a in range(3):
        for b in range(3):
            D[:, a, b] = lam
        D[:, a, a] = lam + 2 * mu
        D[:, 3 + a, 3 + a] = mu
    return D


def stress_from_strain(strain, E, nu):
    """sigma (n,6) = D(E, nu) eps, without forming D."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu)); mu = E / (2 * (1 + nu))
    tr = strain[:, 0] + strain[:, 1] + strain[:, 2]
    s = np.empty_like(strain)
    for a in range(3):
        s[:, a] = lam * tr + 2 * mu * strain[:, a]
        s[:, 3 + a] = mu * strain[:, 3 + a]
    return s


class ElementData3D:
    """Sizes, dof map, center B and the Gauss data of all elements; Ke computed on request."""

    def __init__(self, mesh, gauss_points=8):
        n1 = int(round(gauss_points ** (1.0 / 3)))
        self.gp, self.gw = leggauss(n1)
        self.n1 = n1
        c = mesh.nodes[mesh.conn]                               # (ne,8,3)
        self.size = c.max(axis=1) - c.min(axis=1)               # (ne,3) dx, dy, dz
        self.vol = np.prod(self.size, axis=1)
        self.dof = (3 * mesh.conn[:, :, None] + np.arange(3)[None, None, :]).reshape(-1, 24)
        _, dN = shape_function(0.0, 0.0, 0.0)
        self.Bc = b_matrix(dN, self.size)                       # (ne,6,24)

    def stiffness(self, idx, E, nu, chunk=4000):
        """Ke (len(idx),24,24) for elements idx with moduli E (len(idx),) and Poisson nu."""
        idx = np.asarray(idx); out = np.empty((idx.size, 24, 24))
        D = d_matrix(E, nu)
        for s in range(0, idx.size, chunk):
            sl = slice(s, s + chunk); ii = idx[sl]
            size = self.size[ii]; detJ = self.vol[ii] / 8.0
            K = np.zeros((ii.size, 24, 24))
            for a in range(self.n1):
                for b in range(self.n1):
                    for c in range(self.n1):
                        _, dN = shape_function(self.gp[a], self.gp[b], self.gp[c])
                        B = b_matrix(dN, size)
                        w = self.gw[a] * self.gw[b] * self.gw[c] * detJ
                        DB = np.einsum("nij,njk->nik", D[sl], B)
                        K += np.einsum("nji,njk->nik", B, DB) * w[:, None, None]
            out[sl] = K
        return out
