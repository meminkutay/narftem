"""Run nartfem.axi on one or more model JSON files and print/plot the results.

    python run_nartfem.py examples/FL_AC3_B5_SG4k_GG.json examples/FL_AC3_B5_SG4k_noGG.json --plot

Options
    --plot        modulus / vertical strain / geogrid-confinement maps (PNG next to each JSON)
    --out         also write <json>_nartfem_results.json
    --same-mesh   run all cases on the mesh of the FIRST case (clean with/without comparisons)
    --picard      use the MatFEA update instead of Anderson mixing
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from nartfem.axi import load_model, run
from nartfem.axi import mesh as meshmod


def main():
    p = argparse.ArgumentParser()
    p.add_argument("json", nargs="+")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--out", action="store_true")
    p.add_argument("--same-mesh", action="store_true")
    p.add_argument("--picard", action="store_true")
    a = p.parse_args()
    mesh = None
    table = []
    for f in a.json:
        m = load_model(f)
        if a.picard:
            m.scheme = "picard"
        if a.same_mesh and mesh is None:
            mesh = meshmod.generate(m)
        r = run(m, mesh=mesh)
        base = os.path.splitext(f)[0]
        if a.out:
            r.save(base + "_nartfem_results.json")
        if a.plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            for what in ["Mr", "strain_z"] + (["delsig"] if np.any(r.elem_delsig > 0) else []):
                ax = r.plot(what)
                ax.figure.savefig(base + "_%s.png" % what, dpi=130); plt.close(ax.figure)
        s = r.layer_summary(log=None)
        zt = m.thickness_cum
        et = r.interp("strain_t", 0.0, zt[0] - 0.01) * 1e6
        ez = r.interp("strain_z", 0.0, zt[-1] + 0.01) * 1e6
        table.append((os.path.basename(f), r.iterations, r.converged, s[1]["mean_all"], s[1]["area_near"], et, ez))
    print("\n%-36s %5s %5s %10s %10s %12s %12s" % ("case", "iter", "conv", "E2 mean", "E2 near-A", "eps_t AC bot", "eps_z SG top"))
    for row in table:
        print("%-36s %5d %5s %10.0f %10.0f %12.0f %12.0f" % row)


if __name__ == "__main__":
    main()
