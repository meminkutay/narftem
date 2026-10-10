"""Element-level computations (DefineCalculations.m), vectorized over all elements.

4-node bilinear axisymmetric quad, 2 DOF per node (u_r, u_z), strain vector
[eps_r, eps_z, eps_theta, gamma_rz] (engineering shear), Gauss-Legendre integration,
integration constant = det(J) * w_i * w_j * pi * r   (MatFEA uses pi*r, not 2*pi*r, and
halves the nodal forces accordingly).
"""
from __future__ import annotations
import numpy as np
from numpy.polynomial.legendre import leggauss


def shape_function(xi, eta):
    """N (4,), dN (4,2) = [dN/dxi, dN/deta] - node order: (-1,-1), (1,-1), (1,1), (-1,1)."""
    N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                         (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
    dN = 0.25 * np.array([[-(1 - eta), -(1 - xi)],
                          [+(1 - eta), -(1 + xi)],
                          [+(1 + eta), +(1 + xi)],
                          [-(1 + eta), +(1 - xi)]])
    return N, dN


def b_matrix(N, dN, coords):
    """Strain-displacement matrix for many elements at one natural point.

    coords: (ne, 4, 2) nodal r,z.  Returns B (ne, 4, 8), detJ (ne,), r (ne,)."""
    J = np.einsum("nki,kj->nji", coords, dN)            # J[n] = [[sum r dN/dxi, sum z dN/dxi],[sum r dN/deta, sum z dN/deta]]
    detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    invJ = np.empty_like(J)
    invJ[:, 0, 0] = J[:, 1, 1]; invJ[:, 0, 1] = -J[:, 0, 1]
    invJ[:, 1, 0] = -J[:, 1, 0]; invJ[:, 1, 1] = J[:, 0, 0]
    invJ /= detJ[:, None, None]
    dNdrz = np.einsum("nij,kj->nki", invJ, dN)          # (ne,4,2): dN_k/dr, dN_k/dz
    r = coords[:, :, 0] @ N
    on_axis = r <= 1e-12 * np.max(np.abs(coords[:, :, 0]))
    rs = np.where(on_axis, 1.0, r)
    B = np.zeros((coords.shape[0], 4, 8))
    for k in range(4):
        B[:, 0, 2 * k] = dNdrz[:, k, 0]
        B[:, 1, 2 * k + 1] = dNdrz[:, k, 1]
        B[:, 2, 2 * k] = np.where(on_axis, dNdrz[:, k, 0], N[k] / rs)   # eps_theta = eps_r on the axis
        B[:, 3, 2 * k] = dNdrz[:, k, 1]
        B[:, 3, 2 * k + 1] = dNdrz[:, k, 0]
    return B, detJ, r


def d_matrix(E, nu):
    """Isotropic axisymmetric rigidity matrices (ne, 4, 4)."""
    E = np.asarray(E, float); nu = np.asarray(nu, float)
    D = np.zeros((E.size, 4, 4))
    D[:, 0, 0] = D[:, 1, 1] = D[:, 2, 2] = 1 - nu
    D[:, 0, 1] = D[:, 0, 2] = D[:, 1, 0] = D[:, 1, 2] = D[:, 2, 0] = D[:, 2, 1] = nu
    D[:, 3, 3] = (1 - 2 * nu) / 2
    D *= (E / (1 + nu) / (1 - 2 * nu))[:, None, None]
    return D


class ElementData:
    """Precomputed B matrices, integration constants and edge ("extra") points."""

    def __init__(self, mesh, gauss_points=16):
        coords = mesh.nodes[mesh.conn]                   # (ne,4,2)
        self.coords = coords
        n1 = int(round(np.sqrt(gauss_points)))
        gp, gw = leggauss(n1)
        self.ngp = n1 * n1
        ne = coords.shape[0]
        self.Bg = np.zeros((ne, self.ngp, 4, 8))
        self.cons = np.zeros((ne, self.ngp))
        k = 0
        for i in range(n1):
            for j in range(n1):
                N, dN = shape_function(gp[i], gp[j])
                B, detJ, r = b_matrix(N, dN, coords)
                self.Bg[:, k] = B
                self.cons[:, k] = detJ * gw[i] * gw[j] * np.pi * r
                k += 1
        if np.any(self.cons <= 0):
            raise ValueError("Negative Jacobian in %d elements (check connectivity order)."
                             % np.sum(np.any(self.cons <= 0, axis=1)))
        N, dN = shape_function(0.0, 0.0)
        self.Bc, _, _ = b_matrix(N, dN, coords)           # center B (ne,4,8)
        self.dof = np.empty((ne, 8), int)
        self.dof[:, 0::2] = 2 * mesh.conn
        self.dof[:, 1::2] = 2 * mesh.conn + 1
        self._extra_points(mesh)

    def _extra_points(self, mesh):
        """Edge points of the elements on the model edges and the layer interfaces
        (StiffnessMatrixCalculator BExtra/CoordE): used for plotting/interpolation."""
        coords = self.coords
        W, D = mesh.width, mesh.depth
        tol = 1e-8 * max(W, D)
        self.extra_B, self.extra_xy, self.extra_elem = [], [], []
        r0 = np.any(np.abs(coords[:, :, 0]) <= tol, axis=1)
        rW = np.any(np.abs(coords[:, :, 0] - W) <= tol, axis=1)
        specs = [(r0, (-0.99, 0.0), (-1.0, 0.0)), (rW, (1.0, 0.0), (1.0, 0.0))]
        for mask, (xi, eta), (xic, etac) in specs:
            if not np.any(mask):
                continue
            idx = np.flatnonzero(mask)
            N, dN = shape_function(xi, eta)
            B, _, _ = b_matrix(N, dN, coords[idx])
            Nc, _ = shape_function(xic, etac)
            xy = np.einsum("k,nki->ni", Nc, coords[idx])
            self.extra_B.append(B); self.extra_xy.append(xy); self.extra_elem.append(idx)
        # top / bottom edges of the elements that touch an interface (set later by set_interfaces)
        self._iface_done = False

    def add_interface_points(self, interfaces_depth):
        """Top/bottom edge points of elements touching an interface (positive depths given)."""
        coords = self.coords
        tol = 1e-8 * max(1.0, np.max(np.abs(coords)))
        z_if = -np.asarray(interfaces_depth, float)
        zmax = coords[:, :, 1].max(axis=1); zmin = coords[:, :, 1].min(axis=1)
        top = np.any(np.abs(zmax[:, None] - z_if[None, :]) <= tol, axis=1)
        bot = np.any(np.abs(zmin[:, None] - z_if[None, :]) <= tol, axis=1)
        for mask, (xi, eta) in [(top, (0.0, 1.0)), (bot, (0.0, -1.0))]:
            if not np.any(mask):
                continue
            idx = np.flatnonzero(mask)
            N, dN = shape_function(xi, eta)
            B, _, _ = b_matrix(N, dN, coords[idx])
            xy = np.einsum("k,nki->ni", N, coords[idx])
            self.extra_B.append(B); self.extra_xy.append(xy); self.extra_elem.append(idx)
        self._iface_done = True

    def stiffness(self, D):
        """Element stiffness matrices (ne,8,8) for rigidity matrices D (ne,4,4)."""
        # K = sum_g cons_g * B_g^T D B_g
        DB = np.einsum("nij,ngjk->ngik", D, self.Bg)
        return np.einsum("ngji,ngjk,ng->nik", self.Bg, DB, self.cons)

    def stiffness_subset(self, idx, D):
        DB = np.einsum("nij,ngjk->ngik", D, self.Bg[idx])
        return np.einsum("ngji,ngjk,ng->nik", self.Bg[idx], DB, self.cons[idx])


def overburden(mesh, model_res, thickness):
    """gamma*z at the element centers (GammaZCalculator), 0 for 'Elastic' layers."""
    depth = -mesh.center[:, 1]
    cum = np.concatenate([[0.0], np.cumsum(thickness)])
    gam = model_res["gamma_pci"]
    gz = np.zeros(depth.size)
    nl = len(gam)
    for i in range(nl):
        top = cum[i]
        bot = cum[i + 1] if i < nl - 1 else np.inf
        m = (depth > top) & (depth <= bot) if i > 0 else (depth <= bot)
        gz[m] = np.sum([(cum[k + 1] - cum[k]) * gam[k] for k in range(i)]) + (depth[m] - top) * gam[i]
    for i, mt in enumerate(model_res["mat_type"]):
        if mt == "Elastic":
            gz[mesh.layer == i] = 0.0
    return gz
