"""Mesh- and domain-dependency study of the critical responses.

Part 1: mesh refinement  - Model.mesh_scale multiplies all seed sizes (MatFEA MeshSizeScale;
                           1.3 is the MEAPA default; smaller = finer).
Part 2: domain size      - width (x contact radius) and depth (x pavement thickness).

Tracked:  eps_t at the bottom of the AC (r = 0), eps_z at the top of the subgrade (r = 0) - both as the
          exact FE value in the element (Results.fe_at) and as the MatFEA/MEAPA-style interpolation of
          element-center values (Results.interp),
          surface deflection, near-load area-weighted base modulus, elements, run time.
Writes <case>_mesh_study.png and prints the tables.  Run from PyCharm (CONFIG) or
    python scripts/mesh_study.py examples/FL_AC3_B5_SG4k_GG.json
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from nartfem.axi import load_model, run

# ---------------------------------------------------------------- CONFIG (PyCharm "Run")
CONFIG = dict(
    cases=["examples/FL_AC3_B5_SG4k_noGG.json", "examples/FL_AC3_B5_SG4k_GG.json",
           "examples/EL_AC4_B8_SG15k_GGmid.json"],
    scales=[4.0, 3.0, 2.0, 1.5, 1.3, 1.0, 0.8, 0.6, 0.5],        # part 1
    max_iter=200,
    widths=[20, 40, 80, 160],                                     # part 2: width / contact radius
    depths=[25, 50, 100, 200],                                    # part 2: depth / pavement thickness
    show=False,
)
# ------------------------------------------------------------------------------------


def critical(res, model):
    th = model.thickness_cum
    mesh = res.mesh
    i0 = np.argmin(mesh.nodes[:, 0] ** 2 + mesh.nodes[:, 1] ** 2)
    s = res.layer_summary(log=None)
    nl = [i for i, L in enumerate(model.layers) if L.is_nonlinear]
    return dict(eps_t_ac=res.fe_at("strain_t", 0.0, th[0], side="above") * 1e6,       # exact FE value, AC side
                eps_z_sg=res.fe_at("strain_z", 0.0, th[-1], side="below") * 1e6,      # exact FE value, subgrade side
                eps_t_ac_interp=res.interp("strain_t", 0.0, th[0] - 0.01) * 1e6,       # MatFEA/MEAPA style (center interpolation)
                eps_z_sg_interp=res.interp("strain_z", 0.0, th[-1] + 0.01) * 1e6,
                w0=-res.u[2 * i0 + 1],
                E_near=s[nl[0]]["area_near"] if nl else np.nan,
                ne=mesh.n_elements, it=res.iterations, conv=res.converged)


def sweep(path, key, values, setter):
    rows = []
    for v in values:
        m = load_model(path); m.verbose = False
        m.max_iter = CONFIG.get("max_iter", m.max_iter)
        setter(m, v)
        t = time.time()
        r = run(m)
        c = critical(r, m); c[key] = v; c["t"] = time.time() - t
        rows.append(c)
        print("  %s=%-7g ne=%6d it=%3d conv=%-5s eps_t(AC bot) fe=%7.1f interp=%7.1f | eps_z(SG top) fe=%8.1f interp=%8.1f | w0=%.4f  E_base,near=%6.0f  %.1f s"
              % (key, v, c["ne"], c["it"], c["conv"], c["eps_t_ac"], c["eps_t_ac_interp"], c["eps_z_sg"], c["eps_z_sg_interp"], c["w0"], c["E_near"], c["t"]))
    return rows


def main(cfg):
    import matplotlib
    if not cfg["show"]:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for path in cfg["cases"]:
        name = os.path.splitext(os.path.basename(path))[0]
        print("\n=== %s" % name)
        print("Part 1: mesh refinement (mesh_scale; smaller = finer)")
        P1 = sweep(path, "scale", cfg["scales"], lambda m, v: setattr(m, "mesh_scale", v))
        print("Part 2a: domain width (x contact radius), depth = default 50 x thickness")
        P2 = sweep(path, "width", cfg["widths"], lambda m, v: setattr(m, "width_in", v * m.radius_in))
        print("Part 2b: domain depth (x pavement thickness), width = default 40 a")
        P3 = sweep(path, "depth", cfg["depths"], lambda m, v: setattr(m, "depth_in", v * float(m.thickness_cum[-1])))

        fig, ax = plt.subplots(2, 3, figsize=(15, 7.5))
        # reference = MEAPA defaults: mesh_scale 1.3, width 40 a, depth 50 x thickness
        def refrow(rows, key, val):
            return [r for r in rows if abs(r[key] - val) < 1e-9][0]
        refs = [refrow(P1, "scale", 1.3) if any(abs(r["scale"] - 1.3) < 1e-9 for r in P1) else P1[-1],
                refrow(P2, "width", 40) if any(r["width"] == 40 for r in P2) else P2[0],
                refrow(P3, "depth", 50) if any(r["depth"] == 50 for r in P3) else P3[0]]
        for col, (rows, key, xlabel) in enumerate([(P1, "ne", "number of elements (mesh_scale sweep)"),
                                                   (P2, "width", "domain width / contact radius"),
                                                   (P3, "depth", "domain depth / pavement thickness")]):
            x = [r[key] for r in rows]
            for row, q, lab in [(0, "eps_t_ac", "eps_t, AC bottom"),
                                (1, "eps_z_sg", "eps_z, subgrade top")]:
                y0 = refs[col][q]
                y = [100 * (r[q] / y0 - 1) for r in rows]
                yi = [100 * (r[q + "_interp"] / y0 - 1) for r in rows]
                a = ax[row, col]
                a.axhspan(-2, 2, color="0.9"); a.axhline(0, color="0.5", lw=0.8)
                a.plot(x, yi, "s--", color="0.6", ms=4, label="center interpolation (MatFEA/MEAPA style)")
                a.plot(x, y, "o-", label="exact FE value in the element (fe_at)")
                if row == 0 and col == 0:
                    a.legend(fontsize=7, loc="lower right")
                for xi, yi, r in zip(x, y, rows):
                    if not r["conv"]:
                        a.plot(xi, yi, "rx", ms=10)
                a.set_xscale("log"); a.set_xlabel(xlabel)
                a.set_ylabel("%s: change vs default (%%)\n(default fe_at = %.0f microstrain)" % (lab, y0))
                a.grid(True, alpha=0.3)
                if col == 0:
                    for xi, yi, r in zip(x, y, rows):
                        a.annotate("%g" % r["scale"], (xi, yi), fontsize=7, textcoords="offset points", xytext=(3, 3))
        fig.suptitle("%s - mesh and domain dependency relative to the MEAPA defaults (mesh_scale 1.3, width 40a, depth 50H); labels: mesh_scale; red x: not converged in max_iter; gray band: +/-2%%" % name, fontsize=9)
        fig.tight_layout()
        out = os.path.splitext(path)[0] + "_mesh_study.png"
        fig.savefig(out, dpi=120); print("written:", out)
    if cfg["show"]:
        plt.show()


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    if len(sys.argv) > 1:
        CONFIG["cases"] = sys.argv[1:]
    main(CONFIG)
