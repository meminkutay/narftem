"""One figure with the mesh and the stress / strain / modulus fields of a nartfem.axi run.

    python scripts/plot_fields.py examples/FL_AC3_B5_SG4k_GG.json [--rmax 25] [--zmax 10] [--matlab-mesh]

Writes <json>_fields.png.  --matlab-mesh uses the MatFEA mesh from <json>_matfea_dump.json if present.
Panels: mesh, Mr, geogrid confinement (if any), sigma_z, sigma_r, tau_rz, eps_z, eps_r, eps_theta.
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nartfem.axi import load_model, run
from nartfem.axi import mesh as meshmod


def main():
    p = argparse.ArgumentParser()
    p.add_argument("json"); p.add_argument("--rmax", type=float); p.add_argument("--zmax", type=float)
    p.add_argument("--matlab-mesh", action="store_true"); p.add_argument("--picard", action="store_true")
    a = p.parse_args()
    model = load_model(a.json)
    if a.picard:
        model.scheme = "picard"
    mesh = None
    if a.matlab_mesh:
        dump = os.path.splitext(a.json)[0] + "_matfea_dump.json"
        mesh = meshmod.load(model, dump)
    res = run(model, mesh=mesh)
    rmax = a.rmax or 5 * model.radius_in
    zmax = a.zmax or 1.2 * model.thickness_cum[-1]
    panels = ["mesh", "Mr"] + (["delsig"] if np.any(res.elem_delsig > 0) else []) + \
             ["stress_z", "stress_r", "stress_rz", "strain_z", "strain_r", "strain_t"]
    n = len(panels); ncol = 3; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 3.9 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, what in zip(axes, panels):
        if what == "mesh":
            res.plot_mesh(rmax, zmax, ax=ax)
        else:
            res.plot(what, rmax, zmax, ax=ax, cmap="viridis")
            ax.set_title(what, fontsize=10)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("%s\n%d iterations, converged=%s, layer means (psi): %s" %
                 (model.name, res.iterations, res.converged, ", ".join("%.0f" % e for e in res.E)), fontsize=10)
    fig.tight_layout()
    out = os.path.splitext(a.json)[0] + "_fields.png"
    fig.savefig(out, dpi=120); print("written:", out)


if __name__ == "__main__":
    main()
