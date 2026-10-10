"""Structured hex-8 mesh for the 3D model.

x, y : graded in-plane seeds - uniform ``fine_in`` spacing over the tire zone (all tire
       patches plus half a radius around them), then arithmetic growth (NodeWithBias rule of
       MatFEA) to the outer boundary.
z    : the vertical grading rules of nartfem.axi (layer interfaces and geogrid planes are node
       planes, refinement toward the surface, the interfaces and the geogrid).
Coordinates: x, y in-plane (in), z <= 0 downward (surface at 0), like nartfem.axi.
Node (i, j, k) -> index i + nx*(j + ny*k); element (i, j, k) -> i + (nx-1)*(j + (ny-1)*k).
Hex node order: (-,-,-) (+,-,-) (+,+,-) (-,+,-) (-,-,+) (+,-,+) (+,+,+) (-,+,+) in (xi, eta, zeta),
with zeta = -1 the deeper face.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..axi.mesh import node_with_bias, mesh_size_parameters, _clean


@dataclass
class Mesh3D:
    xs: np.ndarray
    ys: np.ndarray
    zs: np.ndarray               # descending from 0 to -depth
    nodes: np.ndarray            # (nn,3)
    conn: np.ndarray             # (ne,8)
    center: np.ndarray           # (ne,3)
    layer: np.ndarray            # (ne,) 0-based
    source: str = "generated"

    @property
    def n_nodes(self): return self.nodes.shape[0]

    @property
    def n_elements(self): return self.conn.shape[0]

    @property
    def shape(self): return (self.xs.size - 1, self.ys.size - 1, self.zs.size - 1)

    def elem_index(self, i, j, k):
        nx, ny, _ = self.shape
        return i + nx * (j + ny * k)

    def locate(self, x, y, depth):
        """(i, j, k) of the element containing (x, y, depth>0), clipped to the mesh."""
        i = np.clip(np.searchsorted(self.xs, x, side="right") - 1, 0, self.xs.size - 2)
        j = np.clip(np.searchsorted(self.ys, y, side="right") - 1, 0, self.ys.size - 2)
        zq = -np.asarray(depth, float)
        zdesc = -self.zs                                   # ascending depths
        k = np.clip(np.searchsorted(zdesc, -zq, side="right") - 1, 0, self.zs.size - 2)
        return i, j, k


# --------------------------------------------------------------------------- seeds
def inplane_seeds(lo, hi, fine_lo, fine_hi, fine, coarse_max):
    """Nodes on [lo, hi]: uniform ``fine`` on [fine_lo, fine_hi] (clipped to the range), graded
    outward to ``coarse_max`` on both sides."""
    fine_lo, fine_hi = max(lo, fine_lo), min(hi, fine_hi)
    n = max(int(round((fine_hi - fine_lo) / fine)), 1)
    s = np.linspace(fine_lo, fine_hi, n + 1)
    parts = [s]
    if hi - fine_hi > 1e-9:
        g = node_with_bias(hi - fine_hi, fine, coarse_max, "single")
        parts.append(fine_hi + g[1:])
    if fine_lo - lo > 1e-9:
        g = node_with_bias(fine_lo - lo, fine, coarse_max, "single")
        parts.insert(0, fine_lo - g[1:][::-1])
    return _clean(np.concatenate(parts))


def z_seeds(model):
    """Vertical seeds with the nartfem.axi grading rules (scale = model.z_scale)."""
    D = model.depth
    a = model.radius_ref
    P = mesh_size_parameters(40 * a, a, model.z_scale)
    th_cum = np.concatenate([[0.0], model.thickness_cum, [D]])
    gg_depth = []
    for i, L in enumerate(model.layers[:-1]):
        for g in L.geogrids:
            gg_depth.append(th_cum[i + 1] - g.l_in)
    gg_depth = np.array(gg_depth, float)
    crit = _clean(np.concatenate([th_cum, gg_depth]))
    thick = np.diff(crit)
    hs = np.zeros(crit.size); hs[0] = P["ut_min"]
    for ii in range(1, crit.size):
        ec = (2 * a + 4 * crit[ii]) / (2 * a)
        if np.any(np.isclose(crit[ii], gg_depth)):
            ec = max(1.0, ec / 2)
        msc = thick[-1] / 10 if ii == crit.size - 1 else min(thick[ii - 1], thick[ii]) / 10
        if P["ut_min"] * ec > msc:
            ec = max(1.0, msc / P["ut_min"])
        hs[ii] = P["ut_min"] * ec
    z_all = [0.0]
    nsub = crit.size - 1
    for ii in range(nsub):
        top, bot = crit[ii], crit[ii + 1]; t = bot - top
        top_is_gg = np.any(np.isclose(top, gg_depth)); bot_is_gg = np.any(np.isclose(bot, gg_depth))
        if ii == nsub - 1:
            # subgrade: geometric growth (ratio ~1.3) from the spacing at the top of the subgrade,
            # capped at depth/8 - the MatFEA outer-edge rule is too coarse for a 3D stress field
            h0 = 1.5 * hs[ii]; hmax = D / 8.0; ratio = 1.3
            steps = []; h = h0; total = 0.0
            while total < t:
                steps.append(h); total += h; h = min(h * ratio, hmax)
            steps = np.array(steps) * (t / total)
            zz = np.concatenate([[0.0], np.cumsum(steps)])
        elif bot_is_gg:
            up, down = 1.5 * hs[ii], hs[ii + 1]
            if up > down:
                zz = node_with_bias(t, down, up, "single")
                zz = np.concatenate([[0.0], np.cumsum(np.diff(zz)[::-1])]); zz[-1] = t
            else:
                zz = node_with_bias(t, up, down + 1e-3, "single")
        else:
            emin = hs[ii]; emax = 2 * hs[ii + 1] * (0.75 if top_is_gg else 1.0)
            zz = node_with_bias(t, emin, max(emax, emin + 1e-6), "single")
        z_all.extend((top + zz[1:]).tolist())
    z = _clean(z_all); z[-1] = D
    return -z                                               # descending, 0 ... -D


# ----------------------------------------------------------------------- generator
def generate(model) -> Mesh3D:
    a = model.radius_ref
    coarse = max(20.0 * model.coarse_ratio, 4 * model.fine_in)
    # tire zone (full-model coordinates), then clipped by the symmetry range
    tx_lo = min(t.x_in - t.radius_in for t in model.tires) - 0.5 * a
    tx_hi = max(t.x_in + t.radius_in for t in model.tires) + 0.5 * a
    ty_lo = min(t.y_in - t.radius_in for t in model.tires) - 0.5 * a
    ty_hi = max(t.y_in + t.radius_in for t in model.tires) + 0.5 * a
    xs = inplane_seeds(model.x_min, model.x_extent, tx_lo, tx_hi, model.fine_in, coarse)
    ys = inplane_seeds(model.y_min, model.y_extent, ty_lo, ty_hi, model.fine_in, coarse)
    # make sure tire edges/centers are close to node lines (helps the load integration)
    zs = z_seeds(model)
    nx, ny, nz = xs.size, ys.size, zs.size
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")           # (nx,ny,nz)
    nodes = np.column_stack([X.ravel(order="F"), Y.ravel(order="F"), Z.ravel(order="F")])
    idx = np.arange(nx * ny * nz).reshape(nx, ny, nz, order="F")
    i = np.arange(nx - 1)[:, None, None]; j = np.arange(ny - 1)[None, :, None]; k = np.arange(nz - 1)[None, None, :]
    # zeta=-1 face is the deeper one (k+1), zeta=+1 the upper (k)
    conn = np.stack([idx[i, j, k + 1], idx[i + 1, j, k + 1], idx[i + 1, j + 1, k + 1], idx[i, j + 1, k + 1],
                     idx[i, j, k], idx[i + 1, j, k], idx[i + 1, j + 1, k], idx[i, j + 1, k]], axis=-1)
    conn = conn.reshape(-1, 8, order="F")
    center = nodes[conn].mean(axis=1)
    th_cum = np.concatenate([[0.0], model.thickness_cum, [model.depth]])
    layer = np.clip(np.searchsorted(th_cum[1:], -center[:, 2], side="right"), 0, th_cum.size - 2)
    return Mesh3D(xs=xs, ys=ys, zs=zs, nodes=nodes, conn=conn, center=center, layer=layer)
