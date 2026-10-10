"""Results container (the output block of MatFEA.m plus the LayerMrSummary of
DefineNonlinearity.m and interpolants for superposition)."""
from __future__ import annotations
import json
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


class Results:
    """Attributes (all at element centers + edge points, like MatFEA ``out``):
        r, z            coordinates (z positive DOWN, as in MEAPA)
        strain_r/z/t/rz, stress_r/z/t/rz   (tension positive; rz sign flipped like MatFEA)
        Mr              modulus of the element the point belongs to, psi
        layer           1-based layer number
    Element-wise arrays (``elem_*``) for plotting, the mesh, moduli per layer (``E``),
    centerline modulus interpolants (``E_centerline(depth)``), the geogrid confinement
    (``elem_delsig``, ``gg_planes``) and the convergence record."""

    def __init__(self, model, mesh, state, elapsed):
        self.model, self.mesh = model, mesh
        ed = state.ed
        # points: centers then extra (edge) points
        r = [mesh.center[:, 0]]; z = [mesh.center[:, 1]]
        strain = [state.strain]; stress = [state.stress]
        Mr = [state.Mr]; lay = [mesh.layer]
        for B, xy, el in zip(ed.extra_B, ed.extra_xy, ed.extra_elem):
            e_strain = np.einsum("nij,nj->ni", B, state.u[ed.dof[el]])
            e_stress = np.einsum("nij,nj->ni", state.D[el], e_strain)
            r.append(xy[:, 0]); z.append(xy[:, 1]); strain.append(e_strain); stress.append(e_stress)
            Mr.append(state.Mr[el]); lay.append(mesh.layer[el])
        self.r = np.concatenate(r); self.z = -np.concatenate(z)
        strain = np.vstack(strain); stress = np.vstack(stress)
        self.strain_r, self.strain_z, self.strain_t = strain[:, 0], strain[:, 1], strain[:, 2]
        self.strain_rz = -strain[:, 3]
        self.stress_r, self.stress_z, self.stress_t = stress[:, 0], stress[:, 1], stress[:, 2]
        self.stress_rz = -stress[:, 3]
        self.Mr = np.concatenate(Mr); self.layer = np.concatenate(lay) + 1

        # element-wise
        self.elem_center = mesh.center.copy()
        self.elem_Mr = state.Mr.copy()
        self.elem_delsig = state.delsig.copy()
        self.elem_layer = mesh.layer + 1
        self.elem_strain = state.strain.copy(); self.elem_stress = state.stress.copy()
        self.u = state.u.copy()
        self.gg_planes = {k + 1: [dict(z_in=-p["z"], r_in=p["r"], delsig_psi=p["delsig"]) for p in v]
                          for k, v in state.gg_planes.items()}
        self.converged = state.converged; self.iterations = state.iterations
        self.conv_error = state.conv_error; self.elapsed_s = elapsed

        self._layer_moduli()
        self._interp = {}

    # ------------------------------------------------------------- MatFEA out.E / Emodels
    def _layer_moduli(self):
        thc = self.model.thickness_cum
        nl = self.model.n_layers
        self.E = np.zeros(nl)
        self._E_models = []
        for i in range(nl):
            if i == nl - 1:
                m = self.z > thc[-1]
                zmin, zmax = thc[-1], thc[-1] * 100
            elif i == 0:
                m = (self.z > 0) & (self.z <= 0.99 * thc[0])
                zmin, zmax = self.z[m].min(), self.z[m].max()
            else:
                m = (self.z > thc[i - 1]) & (self.z <= thc[i] - (thc[i] - thc[i - 1]) * 0.11)
                zmin, zmax = self.z[m].min(), self.z[m].max()
            self.E[i] = self.Mr[m].mean() if np.any(m) else np.nan
            self._E_models.append(self._centerline_model(zmin, zmax))

    def _centerline_model(self, zmin, zmax):
        mesh = self.mesh
        tol = 1e-8 * mesh.width
        cl_nodes = np.flatnonzero(np.abs(mesh.nodes[:, 0]) <= tol)
        on_cl = np.any(np.isin(mesh.conn, cl_nodes), axis=1)
        data = np.column_stack([-mesh.center[on_cl, 1], self.elem_Mr[on_cl]])
        data = data[np.argsort(data[:, 0])]
        data = np.vstack([[0.0, data[0, 1]], data])
        m = (data[:, 0] >= zmin) & (data[:, 0] <= zmax)
        d = data[m] if np.any(m) else data
        return (d[:, 0], d[:, 1])

    def E_centerline(self, layer_1based, depth):
        """Modulus along the load centerline at the given depth(s) (MatFEA out.Emodels)."""
        zz, EE = self._E_models[layer_1based - 1]
        return np.interp(depth, zz, EE, left=np.nan, right=np.nan)

    # ------------------------------------------------------------------ summary
    def layer_summary(self, near_factor=3.0, log=print):
        """Whole-layer mean | near-load mean (r <= 3a) | near-load area-weighted mean."""
        mesh = self.mesh
        near_r = near_factor * self.model.radius_in
        xy = mesh.nodes[mesh.conn]
        area = 0.5 * np.abs(np.sum(xy[:, :, 0] * np.roll(xy[:, :, 1], -1, axis=1) -
                                   np.roll(xy[:, :, 0], -1, axis=1) * xy[:, :, 1], axis=1))
        rows = []
        if log:
            log("  Layer moduli (psi): whole-layer mean | near-load mean (r <= %.1f in) | near-load area-weighted" % near_r)
        for i, L in enumerate(self.model.layers):
            m = mesh.layer == i
            n = m & (mesh.center[:, 0] <= near_r)
            row = dict(layer=i + 1, name=L.name, mean_all=self.elem_Mr[m].mean(),
                       mean_near=self.elem_Mr[n].mean() if np.any(n) else np.nan,
                       area_near=np.sum(self.elem_Mr[n] * area[n]) / area[n].sum() if np.any(n) else np.nan)
            rows.append(row)
            if log:
                log("    Layer %d %-10s %12.0f | %12.0f | %12.0f" % (i + 1, L.name[:10], row["mean_all"],
                                                                   row["mean_near"], row["area_near"]))
        return rows

    # ------------------------------------------------------------ interpolation
    def interp(self, name, r, z):
        """Interpolate a response (e.g. 'strain_z') at (r, z>0 down) - the Feps_*/Fsig_* of MEAPA."""
        if name not in self._interp:
            pts = np.column_stack([self.r, self.z])
            v = getattr(self, name)
            self._interp[name] = (LinearNDInterpolator(pts, v), NearestNDInterpolator(pts, v))
        lin, near = self._interp[name]
        r = np.asarray(r, float); z = np.asarray(z, float)
        out = lin(r, z)
        bad = ~np.isfinite(out)
        if np.any(bad):
            out[bad] = near(r[bad], z[bad])
        return out

    def response_at(self, r, z):
        """Dict of all strain/stress components at (r, z)."""
        return {k: self.interp(k, r, z) for k in ("strain_r", "strain_z", "strain_t", "strain_rz",
                                                  "stress_r", "stress_z", "stress_t", "stress_rz")}

    # ------------------------------------------------------------------------ I/O
    def to_dict(self):
        return dict(name=self.model.name, converged=bool(self.converged), iterations=int(self.iterations),
                    conv_error=float(self.conv_error), elapsed_s=self.elapsed_s,
                    n_nodes=int(self.mesh.n_nodes), n_elements=int(self.mesh.n_elements),
                    E_layer_mean=self.E.tolist(),
                    layer_summary=self.layer_summary(log=None),
                    elem_center=self.elem_center.tolist(), elem_Mr=self.elem_Mr.tolist(),
                    elem_layer=self.elem_layer.tolist(), elem_delsig=self.elem_delsig.tolist(),
                    elem_strain=self.elem_strain.tolist(), elem_stress=self.elem_stress.tolist(),
                    nodal_disp=self.u.tolist(), gg_planes={str(k): [dict(z_in=float(p["z_in"]),
                                                                        r_in=p["r_in"].tolist(),
                                                                        delsig_psi=p["delsig_psi"].tolist())
                                                                   for p in v] for k, v in self.gg_planes.items()})

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    # ----------------------------------------------------------------------- plots
    def plot(self, what="Mr", rmax=None, zmax=None, ax=None, **kw):
        """Element map of 'Mr', 'delsig', 'strain_z', 'stress_z', ... on the quads.
        For 'Mr' the elastic layers are drawn in gray and the color scale spans the nonlinear layers."""
        import textwrap
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        mesh = self.mesh
        rmax = rmax or 5 * self.model.radius_in
        zmax = zmax or 1.2 * self.model.thickness_cum[-1]
        nonlin = np.array([self.model.layers[i].is_nonlinear for i in mesh.layer])
        if what == "Mr":
            v = self.elem_Mr / 1000.0; label = "Mr (ksi)"
        elif what == "delsig":
            v = self.elem_delsig; label = "geogrid confinement (psi)"
        else:
            comp = {"r": 0, "z": 1, "t": 2, "rz": 3}[what.split("_")[1]]
            v = (self.elem_strain if what.startswith("strain") else self.elem_stress)[:, comp]
            v = v * 1e6 if what.startswith("strain") else v
            label = what + (" (microstrain)" if what.startswith("strain") else " (psi)")
        m = (mesh.center[:, 0] <= rmax) & (-mesh.center[:, 1] <= zmax)
        polys = mesh.nodes[mesh.conn].copy(); polys[:, :, 1] *= -1
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4.8))
        if what in ("Mr", "delsig") and np.any(nonlin):
            gray = m & ~nonlin
            ax.add_collection(PolyCollection(polys[gray], facecolors="0.85", edgecolors="none"))
            sel = m & nonlin
        else:
            sel = m
        pc = PolyCollection(polys[sel], array=v[sel], edgecolors="none", **kw)
        ax.add_collection(pc); ax.set_xlim(0, rmax); ax.set_ylim(zmax, 0)
        ax.set_xlabel("r (in)"); ax.set_ylabel("depth (in)"); ax.set_aspect("equal")
        for zc in self.model.thickness_cum:
            ax.axhline(zc, color="k", lw=0.5)
        for L, planes in self.gg_planes.items():
            for p in planes:
                ax.axhline(p["z_in"], color="r", lw=1.0, ls="--")
        plt.colorbar(pc, ax=ax, label=label, shrink=0.8)
        ax.set_title("\n".join(textwrap.wrap(self.model.name or "nartfem.axi", 70)), fontsize=9)
        return ax
