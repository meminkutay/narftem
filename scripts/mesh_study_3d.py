"""Mesh- and domain-dependency study for nartfem.solid3d (same style as scripts/mesh_study.py).

Sweeps, one at a time from the baseline:
  fine_in   in-plane element size under/around the tires
  z_scale   vertical grading scale (smaller = finer; axi default 1.3, 3D default 3)
  extent    domain width/length as a multiple of the contact radius
  depth     domain depth as a multiple of the pavement thickness
Tracked (exact in-element values, fe_at): AC-bottom max principal strain and subgrade-top eps_z
under a tire and between the tires, mid-base modulus, elements, run time.
Writes <json>_mesh_study3d.png and prints the tables.  Run from PyCharm (CONFIG) or
    python scripts/mesh_study_3d.py examples/FL_AC3_B5_SG4k_GG_dual3d.json
Expect minutes per case with pypardiso, much longer with the CG fallback - start with the
coarse end of the sweeps if time is short.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from nartfem.solid3d import load_model3d, run3d

# ---------------------------------------------------------------- CONFIG (PyCharm "Run")
CONFIG = dict(
    json3d="examples/FL_AC3_B5_SG4k_GG_dual3d.json",
    baseline=dict(fine_in=0.6, z_scale=3.0, extent=30, depth=50),
    fine_in=[1.5, 1.0, 0.75, 0.6, 0.5, 0.4],
    z_scale=[6.0, 4.0, 3.0, 2.0, 1.5],
    extent=[15, 20, 30, 45],
    depth=[25, 50, 100],
    max_iter=40,
    show=False,
)
# ------------------------------------------------------------------------------------


def critical(r, m):
    th = m.thickness_cum
    xt = abs(m.tires[0].x_in)
    def pmax(x, y):
        a = r.fe_at("strain_x", x, y, th[0], "above"); b = r.fe_at("strain_y", x, y, th[0], "above"); g = r.fe_at("strain_xy", x, y, th[0], "above")
        return (0.5 * (a + b) + np.sqrt((0.5 * (a - b)) ** 2 + (0.5 * g) ** 2)) * 1e6
    zmid = 0.5 * (th[0] + th[1])
    return dict(pt_tire=pmax(xt, 0.0), pt_mid=pmax(0.0, 0.0),
                ez_tire=r.fe_at("strain_z", xt, 0.0, th[-1], "below") * 1e6, ez_mid=r.fe_at("strain_z", 0.0, 0.0, th[-1], "below") * 1e6,
                E_tire=r.elem_Mr[r.mesh.elem_index(*r.mesh.locate(xt, 0.0, zmid))],
                ne=r.mesh.n_elements, it=r.iterations, conv=r.converged)


def run_case(path, cfg, **over):
    m = load_model3d(path); m.verbose = False; m.max_iter = cfg["max_iter"]
    p = dict(cfg["baseline"]); p.update(over)
    m.fine_in = p["fine_in"]; m.z_scale = p["z_scale"]
    m.x_max_in = m.y_max_in = p["extent"] * m.radius_ref
    m.depth_in = p["depth"] * float(m.thickness_cum[-1])
    t = time.time(); r = run3d(m); c = critical(r, m); c["t"] = time.time() - t
    return c


def main(cfg):
    import matplotlib
    if not cfg["show"]:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = cfg["json3d"]; name = os.path.splitext(os.path.basename(path))[0]
    print("=== %s   baseline %s" % (name, cfg["baseline"]))
    sweeps = {}
    for key in ("fine_in", "z_scale", "extent", "depth"):
        print("\nsweep %s:" % key)
        rows = []
        for v in cfg[key]:
            c = run_case(path, cfg, **{key: v}); c[key] = v; rows.append(c)
            print("  %s=%-6g ne=%7d it=%3d conv=%-5s | AC-bottom principal: tire %7.1f  mid %7.1f | SG eps_z: tire %8.1f  mid %8.1f | E mid-base tire %6.0f | %.0f s"
                  % (key, v, c["ne"], c["it"], c["conv"], c["pt_tire"], c["pt_mid"], c["ez_tire"], c["ez_mid"], c["E_tire"], c["t"]))
        sweeps[key] = rows
    # figure: % change vs the baseline value of each sweep
    fig, ax = plt.subplots(2, 4, figsize=(18, 8))
    for col, key in enumerate(("fine_in", "z_scale", "extent", "depth")):
        rows = sweeps[key]
        base = [r for r in rows if abs(r[key] - cfg["baseline"][key]) < 1e-9]
        ref = base[0] if base else rows[-1]
        xlab = {"fine_in": "in-plane element size (in)", "z_scale": "z grading scale", "extent": "domain extent / contact radius", "depth": "domain depth / pavement thickness"}[key]
        for row, (qs, lab) in enumerate([(("pt_tire", "pt_mid"), "AC-bottom max principal strain"), (("ez_tire", "ez_mid"), "subgrade-top eps_z")]):
            a = ax[row, col]; a.axhspan(-2, 2, color="0.9"); a.axhline(0, color="0.5", lw=0.8)
            for q, mk, lb in zip(qs, ("o-", "s--"), ("under tire", "between tires")):
                a.plot([r[key] for r in rows], [100 * (r[q] / ref[q] - 1) for r in rows], mk, label=lb)
                for r in rows:
                    if not r["conv"]:
                        a.plot(r[key], 100 * (r[q] / ref[q] - 1), "rx", ms=10)
            a.set_xlabel(xlab); a.set_ylabel("%s: change vs baseline (%%)" % lab); a.grid(True, alpha=0.3)
            if key in ("fine_in", "z_scale"):
                a.invert_xaxis()
            if row == 0 and col == 0:
                a.legend(fontsize=8)
    fig.suptitle("%s - 3D mesh and domain dependency (baseline %s; gray band +/-2 %%; red x: not converged)" % (name, cfg["baseline"]), fontsize=9)
    fig.tight_layout()
    out = os.path.splitext(path)[0] + "_mesh_study3d.png"; fig.savefig(out, dpi=120); print("written:", out)
    if cfg["show"]:
        plt.show()


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    if len(sys.argv) > 1:
        CONFIG["json3d"] = sys.argv[1]
    main(CONFIG)
