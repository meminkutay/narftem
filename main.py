"""nartfem - run one or more model JSON files, print the summary table, write field plots.

Press Run in PyCharm: the settings in CONFIG below are used.
(Command-line arguments, if given, override CONFIG:  python main.py a.json b.json --same-mesh --plot)
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from nartfem.axi import load_model, run
from nartfem.axi import mesh as meshmod

# ------------------------------------------------------------------ CONFIG (edit here)
CONFIG = dict(
    cases=["examples/FL_AC3_B5_SG4k_GG.json",
           "examples/FL_AC3_B5_SG4k_noGG.json"],
    same_mesh=True,      # run all cases on the mesh of the first case (clean with/without comparisons)
    fields=True,         # one figure per case with mesh + Mr + confinement + stresses + strains (<case>_fields.png)
    plot=False,          # also individual maps (<case>_Mr.png, _strain_z.png, _delsig.png)
    out=False,           # write <case>_nartfem_results.json (all element data)
    picard=False,        # True = MatFEA's own update instead of Anderson mixing
    show=True,            # open the figures in a window at the end (PyCharm shows them in the plot tool)
    rmax=None, zmax=None,   # plot window in inches (None = 5a and 1.2 x pavement thickness)
)
# --------------------------------------------------------------------------------------


def plot_fields(res, model, rmax, zmax, out_png):
    import matplotlib.pyplot as plt
    rmax = rmax or 5 * model.radius_in
    zmax = zmax or 1.2 * model.thickness_cum[-1]
    panels = ["mesh", "Mr"] + (["delsig"] if np.any(res.elem_delsig > 0) else []) + \
             ["stress_z", "stress_r", "stress_rz", "strain_z", "strain_r", "strain_t"]
    n = len(panels); ncol = 3; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 3.9 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, what in zip(axes, panels):
        if what == "mesh":
            res.plot_mesh(rmax, zmax, ax=ax)
        else:
            res.plot(what, rmax, zmax, ax=ax, cmap="viridis"); ax.set_title(what, fontsize=10)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("%s\n%d iterations, converged=%s, layer means (psi): %s" %
                 (model.name, res.iterations, res.converged, ", ".join("%.0f" % e for e in res.E)), fontsize=10)
    fig.tight_layout(); fig.savefig(out_png, dpi=120); print("written:", out_png)
    return fig


def main(cfg):
    import matplotlib
    if not cfg["show"]:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mesh = None
    table = []
    for f in cfg["cases"]:
        m = load_model(f)
        if cfg["picard"]:
            m.scheme = "picard"
        if cfg["same_mesh"] and mesh is None:
            mesh = meshmod.generate(m)
        r = run(m, mesh=mesh)
        base = os.path.splitext(f)[0]
        if cfg["out"]:
            r.save(base + "_nartfem_results.json")
        if cfg["fields"]:
            plot_fields(r, m, cfg["rmax"], cfg["zmax"], base + "_fields.png")
        if cfg["plot"]:
            for what in ["Mr", "strain_z"] + (["delsig"] if np.any(r.elem_delsig > 0) else []):
                ax = r.plot(what, cfg["rmax"], cfg["zmax"]); ax.figure.savefig(base + "_%s.png" % what, dpi=130)
        s = r.layer_summary(log=None)
        zt = m.thickness_cum
        et = r.interp("strain_t", 0.0, zt[0] - 0.01) * 1e6
        ez = r.interp("strain_z", 0.0, zt[-1] + 0.01) * 1e6
        table.append((os.path.basename(f), r.iterations, r.converged, s[1]["mean_all"], s[1]["area_near"], et, ez))
    print("\n%-36s %5s %5s %10s %10s %12s %12s" % ("case", "iter", "conv", "E2 mean", "E2 near-A", "eps_t AC bot", "eps_z SG top"))
    for row in table:
        print("%-36s %5d %5s %10.0f %10.0f %12.0f %12.0f" % row)
    if cfg["show"]:
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:                     # command-line use
        p = argparse.ArgumentParser()
        p.add_argument("json", nargs="+"); p.add_argument("--same-mesh", action="store_true")
        p.add_argument("--plot", action="store_true"); p.add_argument("--fields", action="store_true")
        p.add_argument("--out", action="store_true"); p.add_argument("--picard", action="store_true")
        p.add_argument("--show", action="store_true"); p.add_argument("--rmax", type=float); p.add_argument("--zmax", type=float)
        a = p.parse_args()
        CONFIG.update(cases=a.json, same_mesh=a.same_mesh, plot=a.plot, fields=a.fields, out=a.out,
                      picard=a.picard, show=a.show, rmax=a.rmax, zmax=a.zmax)
    main(CONFIG)
