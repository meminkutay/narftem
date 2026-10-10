"""Results of a 3D run: element arrays, exact in-element evaluation, cuts and summaries."""
from __future__ import annotations
import json
import numpy as np
from .elements import shape_function, b_matrix, stress_from_strain

_COMP = {"x": 0, "y": 1, "z": 2, "xy": 3, "yz": 4, "xz": 5}


class Results3D:
    def __init__(self, model, mesh, state, elapsed):
        self.model, self.mesh = model, mesh
        self.u = state.u.copy()
        self.elem_center = mesh.center.copy()
        self.elem_Mr = state.Mr.copy(); self.elem_nu = state.nu.copy()
        self.elem_strain = state.strain.copy(); self.elem_stress = state.stress.copy()
        self.elem_delsig = state.delsig.copy(); self.elem_layer = mesh.layer + 1
        self.gg_planes = {k + 1: v for k, v in state.gg_planes.items()}
        self.converged, self.iterations, self.conv_error = state.converged, state.iterations, state.conv_error
        self.elapsed_s = elapsed
        self._ed = state.ed

    # ---------------------------------------------------------------- evaluation
    def fe_at(self, name, x, y, depth, side="above"):
        """Exact FE value of 'strain_x', 'strain_z', 'stress_xy', ... at (x, y, depth>0).  On a
        layer interface side='above'/'below' selects the element.  Full-model coordinates: with
        symmetry, |x| / |y| are used."""
        x = np.atleast_1d(np.asarray(x, float)); y = np.atleast_1d(np.asarray(y, float)); depth = np.atleast_1d(np.asarray(depth, float))
        x, y, depth = np.broadcast_arrays(x, y, depth)
        if self.model.symmetry in ("quarter", "half_x"):
            x = np.abs(x)
        if self.model.symmetry in ("quarter", "half_y"):
            y = np.abs(y)
        mesh = self.mesh; ed = self._ed
        comp = _COMP[name.split("_")[1]]
        out = np.empty(x.size)
        zdesc = -mesh.zs
        for n in range(x.size):
            i = int(np.clip(np.searchsorted(mesh.xs, x[n], side="right") - 1, 0, mesh.xs.size - 2))
            j = int(np.clip(np.searchsorted(mesh.ys, y[n], side="right") - 1, 0, mesh.ys.size - 2))
            on = np.flatnonzero(np.isclose(zdesc, depth[n]))
            if on.size and side == "above":
                k = int(np.clip(on[0] - 1, 0, mesh.zs.size - 2))
            elif on.size:
                k = int(np.clip(on[0], 0, mesh.zs.size - 2))
            else:
                k = int(np.clip(np.searchsorted(zdesc, depth[n], side="right") - 1, 0, mesh.zs.size - 2))
            e = mesh.elem_index(i, j, k)
            c = mesh.nodes[mesh.conn[e]]
            lo, hi = c.min(axis=0), c.max(axis=0)
            xi = 2 * (np.array([x[n], y[n], -depth[n]]) - lo) / (hi - lo) - 1
            xi = np.clip(xi, -1, 1)
            _, dN = shape_function(*xi)
            B = b_matrix(dN, ed.size[e:e + 1])[0]
            strain = B @ self.u[ed.dof[e]]
            val = strain if name.startswith("strain") else stress_from_strain(strain[None], self.elem_Mr[e:e + 1], self.elem_nu[e:e + 1])[0]
            out[n] = val[comp]
        return out if out.size > 1 else out[0]

    # ------------------------------------------------------------------ summaries
    def layer_summary(self, near_factor=3.0, log=print):
        mesh = self.mesh; a = self.model.radius_ref
        vol = self._ed.vol
        c = mesh.center
        dist = np.min([np.hypot(c[:, 0] - t.x_in, c[:, 1] - t.y_in) for t in self.model.tires], axis=0)
        rows = []
        if log:
            log("  Layer moduli (psi): whole-layer volume mean | near-load (within %.1f in of a tire) volume mean" % (near_factor * a))
        for i, L in enumerate(self.model.layers):
            m = mesh.layer == i; n = m & (dist <= near_factor * a)
            row = dict(layer=i + 1, name=L.name, mean_all=np.sum(self.elem_Mr[m] * vol[m]) / vol[m].sum(),
                       near=np.sum(self.elem_Mr[n] * vol[n]) / vol[n].sum() if np.any(n) else np.nan)
            rows.append(row)
            if log:
                log("    Layer %d %-10s %12.0f | %12.0f" % (i + 1, L.name[:10], row["mean_all"], row["near"]))
        return rows

    def to_dict(self):
        return dict(name=self.model.name, converged=bool(self.converged), iterations=int(self.iterations),
                    conv_error=float(self.conv_error), elapsed_s=self.elapsed_s, n_nodes=int(self.mesh.n_nodes),
                    n_elements=int(self.mesh.n_elements), layer_summary=self.layer_summary(log=None),
                    elem_center=self.elem_center.tolist(), elem_Mr=self.elem_Mr.tolist(), elem_layer=self.elem_layer.tolist(),
                    elem_delsig=self.elem_delsig.tolist(), elem_strain=self.elem_strain.tolist(), elem_stress=self.elem_stress.tolist())

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    # ----------------------------------------------------------------------- plots
    def plot_cut(self, what="Mr", plane="xz", at=0.0, xmax=None, zmax=None, ax=None, **kw):
        """Vertical cut: plane='xz' at y=at (through the dual tires) or 'yz' at x=at; or a
        plan view plane='xy' at depth=at.  what: 'Mr', 'delsig', 'strain_z', 'stress_x', ..."""
        import textwrap
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        mesh = self.mesh; m = self.model
        nonlin = np.array([m.layers[i].is_nonlinear for i in mesh.layer])
        if what == "Mr":
            v = self.elem_Mr / 1000.0; label = "Mr (ksi)"
        elif what == "delsig":
            v = self.elem_delsig; label = "geogrid confinement (psi)"
        else:
            comp = _COMP[what.split("_")[1]]
            v = (self.elem_strain if what.startswith("strain") else self.elem_stress)[:, comp]
            v = v * 1e6 if what.startswith("strain") else v
            label = what + (" (microstrain)" if what.startswith("strain") else " (psi)")
        xs, ys, zs = mesh.xs, mesh.ys, mesh.zs
        if plane == "xz":
            j = int(np.clip(np.searchsorted(ys, abs(at), side="right") - 1, 0, ys.size - 2))
            sel = np.flatnonzero((mesh.center[:, 1] >= ys[j]) & (mesh.center[:, 1] <= ys[j + 1]))
            h, vcoord = 0, 2; hl, vl = "x (in)", "depth (in)"
        elif plane == "yz":
            i = int(np.clip(np.searchsorted(xs, abs(at), side="right") - 1, 0, xs.size - 2))
            sel = np.flatnonzero((mesh.center[:, 0] >= xs[i]) & (mesh.center[:, 0] <= xs[i + 1]))
            h, vcoord = 1, 2; hl, vl = "y (in)", "depth (in)"
        else:
            zd = -zs; k = int(np.clip(np.searchsorted(zd, at, side="right") - 1, 0, zs.size - 2))
            sel = np.flatnonzero((-mesh.center[:, 2] >= zd[k]) & (-mesh.center[:, 2] <= zd[k + 1]))
            h, vcoord = 0, 1; hl, vl = "x (in)", "y (in)"
        hmax = xmax or (max(abs(t.x_in) for t in m.tires) + 4 * m.radius_ref)
        vmax = zmax or (1.2 * m.thickness_cum[-1] if plane != "xy" else hmax)
        c = mesh.nodes[mesh.conn[sel]]
        lo, hi = c.min(axis=1), c.max(axis=1)
        if plane != "xy":
            polys = np.stack([np.column_stack([lo[:, h], -hi[:, 2]]), np.column_stack([hi[:, h], -hi[:, 2]]),
                              np.column_stack([hi[:, h], -lo[:, 2]]), np.column_stack([lo[:, h], -lo[:, 2]])], axis=1)
            keep = (mesh.center[sel, h] <= hmax) & (-mesh.center[sel, 2] <= vmax)
        else:
            polys = np.stack([np.column_stack([lo[:, 0], lo[:, 1]]), np.column_stack([hi[:, 0], lo[:, 1]]),
                              np.column_stack([hi[:, 0], hi[:, 1]]), np.column_stack([lo[:, 0], hi[:, 1]])], axis=1)
            keep = (mesh.center[sel, 0] <= hmax) & (mesh.center[sel, 1] <= vmax)
        sel = sel[keep]; polys = polys[keep]
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4.8))
        if what in ("Mr", "delsig"):
            g = ~nonlin[sel]
            ax.add_collection(PolyCollection(polys[g], facecolors="0.85", edgecolors="none"))
            pc = PolyCollection(polys[~g], array=v[sel][~g], edgecolors="none", **kw)
        else:
            pc = PolyCollection(polys, array=v[sel], edgecolors="none", **kw)
        ax.add_collection(pc)
        if plane != "xy":
            ax.set_xlim(mesh.xs[0] if plane == "xz" else mesh.ys[0], hmax); ax.set_ylim(vmax, 0)
            for zc in m.thickness_cum:
                ax.axhline(zc, color="k", lw=0.5)
            for t in m.tires:
                cc = t.x_in if plane == "xz" else t.y_in
                ax.plot([cc - t.radius_in, cc + t.radius_in], [0, 0], "r-", lw=3)
        else:
            ax.set_xlim(mesh.xs[0], hmax); ax.set_ylim(mesh.ys[0], vmax)
            for t in m.tires:
                ax.add_patch(plt.Circle((t.x_in, t.y_in), t.radius_in, fill=False, color="r"))
        ax.set_xlabel(hl); ax.set_ylabel(vl); ax.set_aspect("equal")
        plt.colorbar(pc, ax=ax, label=label, shrink=0.8)
        ax.set_title("\n".join(textwrap.wrap("%s  [%s cut at %g]" % (m.name or "nartfem.solid3d", plane, at), 80)), fontsize=9)
        return ax
