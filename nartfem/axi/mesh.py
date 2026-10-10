"""Mesh generation and import (DefineGenerateMesh.m / DefineLoadMesh.m).

The MATLAB mesher is a recursive advancing-front quad generator.  The Python mesher
produces a *structured* graded mesh instead: the same seed-node grading rules as
MatFEA (NodeWithBias, expansion of the fine zone with depth, refinement at the
geogrid plane), but the radial seeds of the surface are used at every depth so the
mesh is a simple tensor product and always conforming.  Element counts are of the
same order as MatFEA (a few thousand), run times are seconds.

For a mesh-identical comparison with MATLAB, a MatFEA mesh can be imported from
    * a MatFEA '.mesh' text file (see DefineLoadMesh.m / Template.mesh), or
    * a JSON dump written by matlab/run_matfea_dump.m (keys Nodes, Connectivity).

Coordinates: column 0 = r (>= 0), column 1 = z (<= 0, surface at 0) - MatFEA convention.
Connectivity: 4 nodes per element, counter-clockwise starting at the lower-left corner
(down-left, down-right, up-right, up-left), 0-based.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
import numpy as np


@dataclass
class Mesh:
    nodes: np.ndarray          # (nn, 2)  r, z
    conn: np.ndarray           # (ne, 4)  0-based
    center: np.ndarray         # (ne, 2)
    layer: np.ndarray          # (ne,)    0-based layer index
    width: float
    depth: float
    source: str = "generated"

    @property
    def n_nodes(self): return self.nodes.shape[0]

    @property
    def n_elements(self): return self.conn.shape[0]


# --------------------------------------------------------------------------- seeds
def node_with_bias(length, emin, emax, kind="single"):
    """Port of NodeWithBias.m: arithmetic-sequence node spacing from emin to emax."""
    if length <= 0:
        return np.array([])
    if kind == "single":
        N = int(np.ceil(length / (emin + (emax - emin) / 2))) - 1
        d = 2 * ((length / (N + 1)) - emin) / N if N > 0 else 0.0
        if N == 0:
            d = 0.0
        elif d <= 0:
            N = N - 1 if N > 1 else 1
            d = 2 * ((length / (N + 1)) - emin) / N
        steps = emin + d * np.arange(N + 1)
        res = np.concatenate([[0.0], np.cumsum(steps)])
        res[-1] = length
    else:                                           # 'double': fine at both ends
        temp = (length + emax) / 2
        N = int(np.ceil(temp / (emin + (emax - emin) / 2))) - 1
        N = max(N, 1)
        d = 2 * ((temp / (N + 1)) - emin) / N
        steps = emin + d * np.arange(N + 1)
        res = np.concatenate([[0.0], np.cumsum(steps)])[:-1]
        res = np.concatenate([res, length - res[::-1]])
    if res.size < 2:
        res = np.array([0.0, length])
    return res


def mesh_size_parameters(width, radius, scale):
    """Port of SideFunc03MeshProperties.m."""
    wf = width / 200.0
    return dict(ut_min=0.1 * scale, ut_max=0.3 * scale,
                oe_max=20.0 * scale * wf, oe_min=10.0 * wf,
                be_min=20.0 * scale, be_max=50.0 * scale, tire=radius)


def _clean(seq, tol=1e-9):
    seq = np.unique(np.round(np.asarray(seq, float), 10))
    keep = np.concatenate([[True], np.diff(seq) > tol])
    return seq[keep]


# ----------------------------------------------------------------------- generator
def generate(model) -> Mesh:
    """Structured graded mesh following the MatFEA seed rules."""
    W, D, a = model.width, model.depth, model.radius_in
    P = mesh_size_parameters(W, a, model.mesh_scale)
    th_cum = np.concatenate([[0.0], model.thickness_cum, [D]])     # positive depths of interfaces

    # critical depths: layer interfaces + geogrid planes (positive, downward)
    gg_depth = []
    for i, L in enumerate(model.layers[:-1]):
        for g in L.geogrids:
            gg_depth.append(th_cum[i + 1] - g.l_in)
    gg_depth = np.array(gg_depth, float)
    crit = _clean(np.concatenate([th_cum, gg_depth]))
    thick = np.diff(crit)

    # ---- radial seeds on the surface (Horizontal{1})
    c1 = node_with_bias(a, P["ut_min"], P["ut_max"], "double")
    c2 = node_with_bias(a, P["ut_min"], P["ut_max"] * 2, "single")
    c3 = node_with_bias(W - 2 * a, P["ut_max"] * 2, P["oe_min"], "single")
    r = _clean(np.concatenate([c1, a + c2[1:], 2 * a + c3[1:]]))
    r[0], r[-1] = 0.0, W

    # ---- first radial spacing at every critical depth (the "ExpandCoeff" rule)
    hs = np.zeros(crit.size)
    hs[0] = P["ut_min"]
    for ii in range(1, crit.size):
        ec = (2 * a + 4 * crit[ii]) / (2 * a)
        is_gg = np.any(np.isclose(crit[ii], gg_depth))
        if is_gg:
            ec = max(1.0, ec / 2)
        msc = thick[-1] / 10 if ii == crit.size - 1 else min(thick[ii - 1], thick[ii]) / 10
        if P["ut_min"] * ec > msc:
            ec = max(1.0, msc / P["ut_min"])
        hs[ii] = P["ut_min"] * ec

    # ---- vertical seeds per sub-layer (inner-edge rules of DefineGenerateMesh)
    z_all = [0.0]
    nsub = crit.size - 1
    for ii in range(nsub):
        top, bot = crit[ii], crit[ii + 1]
        t = bot - top
        top_is_gg = np.any(np.isclose(top, gg_depth))
        bot_is_gg = np.any(np.isclose(bot, gg_depth))
        if ii == nsub - 1:                                           # subgrade block
            emin = 1.5 * hs[ii]
            emax = 1.5 * P["be_min"]
            zz = node_with_bias(t, emin, max(emax, emin + 1e-6), "single")
        elif bot_is_gg:                                              # refine toward the geogrid below
            inner_up = 1.5 * hs[ii]
            inner_down = hs[ii + 1]
            if inner_up > inner_down:
                zz = node_with_bias(t, inner_down, inner_up, "single")
                zz = np.concatenate([[0.0], np.cumsum(np.diff(zz)[::-1])]); zz[-1] = t
            else:
                zz = node_with_bias(t, inner_up, inner_down + 1e-3, "single")
        else:
            emin = hs[ii]
            emax = 2 * hs[ii + 1] * (0.75 if top_is_gg else 1.0)
            zz = node_with_bias(t, emin, max(emax, emin + 1e-6), "single")
        z_all.extend((top + zz[1:]).tolist())
    z = _clean(z_all)
    z[-1] = D
    zneg = -z                                                        # MatFEA: z <= 0

    # ---- tensor-product nodes and elements
    nr, nz = r.size, z.size
    R, Z = np.meshgrid(r, zneg, indexing="ij")                      # node (i, j) -> i + j*nr
    nodes = np.column_stack([R.ravel(order="F"), Z.ravel(order="F")])
    idx = np.arange(nr * nz).reshape(nr, nz, order="F")
    i = np.arange(nr - 1)[:, None]
    j = np.arange(nz - 1)[None, :]
    conn = np.stack([idx[i, j + 1], idx[i + 1, j + 1], idx[i + 1, j], idx[i, j]], axis=-1)
    conn = conn.reshape(-1, 4)
    center = nodes[conn].mean(axis=1)
    layer = _layer_of(center[:, 1], th_cum)
    return Mesh(nodes=nodes, conn=conn, center=center, layer=layer, width=W, depth=D)


def _layer_of(zc, th_cum):
    """0-based layer index from the element center depth (MergeAllNodes rule)."""
    depth = -np.asarray(zc)
    lay = np.searchsorted(th_cum[1:], depth, side="right")          # first k with depth < cum[k]
    return np.clip(lay, 0, th_cum.size - 2)


# -------------------------------------------------------------------------- import
def load(model, path: str) -> Mesh:
    """Import a MatFEA mesh ('.mesh' text or MATLAB dump JSON). Width/depth come from the mesh."""
    if path.lower().endswith(".json"):
        with open(path) as f:
            d = json.load(f)
        d = d.get("mesh", d)
        nodes = np.asarray(d["Nodes"], float)
        conn = np.asarray(d["Connectivity"], int) - 1
    else:
        nodes, conn = _read_mesh_text(path)
    return _finish_import(model, nodes, conn, path)


def from_arrays(model, nodes, conn_1based) -> Mesh:
    return _finish_import(model, np.asarray(nodes, float), np.asarray(conn_1based, int) - 1, "arrays")


def _finish_import(model, nodes, conn, source):
    W = float(nodes[:, 0].max()); D = float(-nodes[:, 1].min())
    conn = _orient(nodes, conn)
    center = nodes[conn].mean(axis=1)
    th_cum = np.concatenate([[0.0], model.thickness_cum, [D]])
    layer = _layer_of(center[:, 1], th_cum)
    return Mesh(nodes=nodes, conn=conn, center=center, layer=layer, width=W, depth=D, source=source)


def _read_mesh_text(path):
    nodes, elems = [], []
    mode = "nodes"
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    for ln in lines[1:]:
        if ln.lower().startswith("elements"):
            mode = "elements"; continue
        vals = [float(v) for v in ln.replace(",", " ").split()]
        if mode == "nodes":
            nodes.append(vals[1:3])
        else:
            elems.append([int(v) for v in vals[1:5]])
    return np.asarray(nodes, float), np.asarray(elems, int) - 1


def _orient(nodes, conn):
    """Make every element counter-clockwise (positive Jacobian) and start at the lower-left node."""
    conn = conn.copy()
    p = nodes[conn]                                                  # (ne,4,2)
    area = 0.5 * np.sum(p[:, :, 0] * np.roll(p[:, :, 1], -1, axis=1) -
                        np.roll(p[:, :, 0], -1, axis=1) * p[:, :, 1], axis=1)
    conn[area < 0] = conn[area < 0][:, ::-1]
    p = nodes[conn]
    # rotate so that node 0 is the one with the smallest (z, r) -> lower-left
    key = p[:, :, 1] * 1e6 + p[:, :, 0]
    start = np.argmin(key, axis=1)
    rows = np.arange(conn.shape[0])[:, None]
    conn = conn[rows, (start[:, None] + np.arange(4)[None, :]) % 4]
    return conn


def relabel(model, mesh) -> Mesh:
    """Copy of a mesh with the element layer numbers recomputed for ``model`` (same thicknesses)."""
    th_cum = np.concatenate([[0.0], model.thickness_cum, [mesh.depth]])
    return Mesh(nodes=mesh.nodes, conn=mesh.conn, center=mesh.center,
                layer=_layer_of(mesh.center[:, 1], th_cum), width=mesh.width, depth=mesh.depth,
                source=mesh.source)
