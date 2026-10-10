"""Global assembly, tire loads, boundary conditions and solver backends for the 3D model."""
from __future__ import annotations
import time
import warnings
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, cg, LinearOperator
from numpy.polynomial.legendre import leggauss


def assemble(Ke, dof, ndof):
    rows = np.repeat(dof, 24, axis=1).ravel()
    cols = np.tile(dof, (1, 24)).ravel()
    return sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(ndof, ndof)).tocsr()


# ------------------------------------------------------------------------ loads
def tire_loads(mesh, model):
    """Consistent nodal loads of the tire pressure patches on the surface faces (z = 0).
    Each surface face is integrated with 4x4 Gauss points; the pressure is applied where the
    point lies inside a tire patch.  The result is rescaled per tire to the exact load that
    falls inside the modeled domain (symmetry)."""
    xs, ys = mesh.xs, mesh.ys
    nx, ny = xs.size, ys.size                      # node counts (row stride of the node numbering)
    gp, gw = leggauss(4)
    F = np.zeros(3 * mesh.n_nodes)
    for t in model.tires:
        Ft = np.zeros_like(F)
        # candidate surface faces: those overlapping the tire bounding box
        r = t.radius_in if t.shape == "circle" else t.half_side_in * np.sqrt(2)
        i0 = max(np.searchsorted(xs, t.x_in - r) - 1, 0); i1 = min(np.searchsorted(xs, t.x_in + r) + 1, nx - 1)
        j0 = max(np.searchsorted(ys, t.y_in - r) - 1, 0); j1 = min(np.searchsorted(ys, t.y_in + r) + 1, ny - 1)
        for i in range(i0, i1):
            for j in range(j0, j1):
                x0, x1 = xs[i], xs[i + 1]; y0, y1 = ys[j], ys[j + 1]
                nodes = [i + nx * j, i + 1 + nx * j, i + 1 + nx * (j + 1), i + nx * (j + 1)]   # k = 0 (surface)
                fn = np.zeros(4)
                for a in range(4):
                    for b in range(4):
                        x = 0.5 * (x0 + x1) + 0.5 * (x1 - x0) * gp[a]
                        y = 0.5 * (y0 + y1) + 0.5 * (y1 - y0) * gp[b]
                        if t.shape == "circle":
                            inside = (x - t.x_in) ** 2 + (y - t.y_in) ** 2 <= t.radius_in ** 2
                        else:
                            inside = abs(x - t.x_in) <= t.half_side_in and abs(y - t.y_in) <= t.half_side_in
                        if not inside:
                            continue
                        N = 0.25 * np.array([(1 - gp[a]) * (1 - gp[b]), (1 + gp[a]) * (1 - gp[b]),
                                             (1 + gp[a]) * (1 + gp[b]), (1 - gp[a]) * (1 + gp[b])])
                        fn += N * t.pressure_psi * gw[a] * gw[b] * 0.25 * (x1 - x0) * (y1 - y0)
                for n, f in zip(nodes, fn):
                    Ft[3 * n + 2] -= f
        # exact load inside the domain (fraction of the patch area in the modeled quadrant)
        frac = _patch_fraction(t, model)
        target = -t.load_lb * frac
        if Ft.sum() != 0:
            Ft *= target / Ft.sum()
        F += Ft
    return F


def _patch_fraction(t, model):
    """Fraction of the tire patch inside the modeled domain (x >= x_min, y >= y_min)."""
    n = 400
    r = t.radius_in if t.shape == "circle" else t.half_side_in
    xx = t.x_in + np.linspace(-r, r, n); yy = t.y_in + np.linspace(-r, r, n)
    X, Y = np.meshgrid(xx, yy, indexing="ij")
    if t.shape == "circle":
        inside = (X - t.x_in) ** 2 + (Y - t.y_in) ** 2 <= r ** 2
    else:
        inside = np.ones_like(X, bool)
    dom = (X >= model.x_min - 1e-9) & (Y >= model.y_min - 1e-9)
    return float((inside & dom).sum()) / float(inside.sum())


# -------------------------------------------------------------------- boundaries
def fixed_dofs(mesh, model):
    nodes = mesh.nodes
    tol = 1e-9 * max(model.x_extent, model.y_extent, model.depth)
    out = []
    # symmetry / outer planes: normal displacement fixed (rollers); bottom: all fixed
    for coord, value in [(0, mesh.xs[0]), (0, mesh.xs[-1]), (1, mesh.ys[0]), (1, mesh.ys[-1])]:
        idx = np.flatnonzero(np.abs(nodes[:, coord] - value) <= tol)
        out.append(3 * idx + coord)
    idx = np.flatnonzero(np.abs(nodes[:, 2] - mesh.zs[-1]) <= tol)
    out.extend([3 * idx, 3 * idx + 1, 3 * idx + 2])
    return np.unique(np.concatenate(out))


# ------------------------------------------------------------------------ solver
class Solver3D:
    """backend: 'pardiso' (pypardiso), 'pyamg' (smoothed aggregation + CG), 'splu' (SuperLU),
    'cg' (Jacobi-preconditioned CG, warm started).  'auto' picks the best available."""

    def __init__(self, ndof, fixed, backend="auto", log=None):
        self.ndof = ndof
        self.free = np.setdiff1d(np.arange(ndof), fixed)
        self.log = log or (lambda *a: None)
        self.x_prev = None
        if backend == "auto":
            backend = "pardiso" if _has("pypardiso") else ("pyamg" if _has("pyamg") else
                      ("splu" if self.free.size <= 15000 else "cg"))
        self.backend = backend
        self.log("  - solver backend: %s (%d free dofs)" % (backend, self.free.size))

    def solve(self, K, F):
        Kr = K[self.free][:, self.free].tocsc()
        b = F[self.free]
        t0 = time.time()
        if self.backend == "pardiso":
            import pypardiso
            x = pypardiso.spsolve(Kr.tocsr(), b)
        elif self.backend == "pyamg":
            import pyamg
            ml = pyamg.smoothed_aggregation_solver(Kr.tocsr(), B=_rigid_body_modes(self), max_coarse=500)
            x = ml.solve(b, x0=self.x_prev, tol=1e-8, accel="cg", maxiter=400)
        elif self.backend == "splu":
            lu = splu(Kr, permc_spec="NATURAL", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
            x = lu.solve(b)
        else:
            d = Kr.diagonal(); M = LinearOperator(Kr.shape, matvec=lambda v: v / d)
            x, info = cg(Kr, b, x0=self.x_prev, M=M, rtol=1e-7, maxiter=30000)
            if info != 0:
                warnings.warn("CG did not converge (info=%d)" % info)
        self.x_prev = x
        self.last_time = time.time() - t0
        u = np.zeros(self.ndof); u[self.free] = x
        return u


def _has(mod):
    try:
        __import__(mod); return True
    except Exception:
        return False


def _rigid_body_modes(solver):
    # placeholder near-nullspace: translations only (AMG still converges with CG acceleration)
    n = solver.free.size
    B = np.zeros((n, 3))
    comp = solver.free % 3
    for c in range(3):
        B[comp == c, c] = 1.0
    return B
