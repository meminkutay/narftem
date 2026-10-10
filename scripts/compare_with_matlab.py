"""Compare nartfem.axi with a MatFEA (MATLAB) run on the identical mesh.

Steps
  1. MATLAB:  >> run_matfea_dump('examples/FL_AC3_B5_SG4k_GG.json')
              -> examples/FL_AC3_B5_SG4k_GG_matfea_dump.json
  2. Python:  $ python scripts/compare_with_matlab.py examples/FL_AC3_B5_SG4k_GG.json

What is compared (same mesh, same inputs):
  a. first linear solve: nodal displacements           -> should agree to ~1e-8 (solver round-off)
  b. moduli after the first update (DefineResponses)   -> should agree to ~1e-6 (same stresses, same
                                                           Mr model, same geogrid equation)
  c. converged moduli / strains                        -> Picard scheme reproduces MatFEA's iteration
                                                           path; Anderson scheme gives the fixed point
                                                           (differences appear where MatFEA oscillates)
A figure with the comparisons is written next to the dump.
"""
from __future__ import annotations
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nartfem.axi import load_model, run                       # noqa: E402
from nartfem.axi import mesh as meshmod                        # noqa: E402
from nartfem.axi.elements import ElementData, overburden       # noqa: E402
from nartfem.axi.assembly import force_vector, fixed_dofs, Solver   # noqa: E402
from nartfem.axi.responses import State                        # noqa: E402


def rel(a, b, floor=1e-12):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return np.abs(a - b) / np.maximum(np.abs(b), floor * np.max(np.abs(b)) + floor)


def report(label, a, b, scale_floor=1e-3):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    ref = np.max(np.abs(b))
    d = np.abs(a - b)
    big = np.abs(b) > scale_floor * ref
    r = d[big] / np.abs(b[big]) if np.any(big) else np.array([0.0])
    print("  %-38s max|diff|/max|ref| = %.2e   median rel = %.2e   max rel = %.2e  (n=%d)"
          % (label, d.max() / ref, np.median(r), r.max(), a.size))
    return d.max() / ref


def main(json_path, dump_path=None):
    if dump_path is None:
        dump_path = os.path.splitext(json_path)[0] + "_matfea_dump.json"
    with open(dump_path) as f:
        dump = json.load(f)
    model = load_model(json_path)
    model.verbose = False
    nodes = np.asarray(dump["mesh"]["Nodes"], float)
    conn = np.asarray(dump["mesh"]["Connectivity"], int)
    mesh = meshmod.from_arrays(model, nodes, conn)
    mesh.layer = np.asarray(dump["mesh"]["EleLyrNum"], int).ravel() - 1
    mesh.source = "MatFEA mesh (%s)" % os.path.basename(dump_path)
    print("Model: %s\nMesh : %d nodes, %d elements from MATLAB" % (model.name, mesh.n_nodes, mesh.n_elements))

    # ---- a + b: linear solve and first modulus update, step by step
    res = model.resolved()
    ed = ElementData(mesh, model.gauss_points)
    ed.add_interface_points(np.concatenate([[0.0], model.thickness_cum, [mesh.depth]]))
    st = State(model, mesh, ed, res)
    st.gz = overburden(mesh, res, model.thickness)
    F = force_vector(mesh, model.pressure_psi, model.radius_in)
    solver = Solver(2 * mesh.n_nodes, fixed_dofs(mesh))
    st.u = solver.solve(st.K, F)
    print("\n[a] first linear solve")
    report("seed moduli", st.Mr, dump["linear"]["elem_Mr_seed"])
    report("total nodal force", [F.sum()], [dump["linear"]["force_total"]])
    report("nodal displacements", st.u, dump["linear"]["nodal_disp"])
    st.responses()
    print("\n[b] first modulus update (DefineResponses + DefineGeogrid)")
    it1 = dump["iteration1"]
    report("element strains", st.strain, it1["elem_strain"])
    report("element stresses", st.stress, it1["elem_stress"])
    report("moduli after update 1", st.Mr, it1["elem_Mr"])
    if np.any(np.asarray(it1["elem_delsig"]) != 0):
        report("geogrid confinement delSig", -st.delsig, it1["elem_delsig"])
    report("MrError", st.MrError, it1["MrError"])

    # ---- c: converged solutions
    fin = dump["final"]
    print("\n[c] converged solution   (MATLAB: converged=%s, %s iterations, error %.2f%%)"
          % (fin["converged"], fin["iterations"], 100 * float(fin["conv_error"])))
    out = {}
    for scheme in ("picard", "anderson"):
        model.scheme = scheme
        r = run(model, mesh=mesh)
        out[scheme] = r
        print("  nartfem %-8s converged=%s, %d iterations, error %.2f%%" % (scheme, r.converged, r.iterations, 100 * r.conv_error))
        report("   moduli", r.elem_Mr, fin["elem_Mr"])
        report("   strains", r.elem_strain, fin["elem_strain"])
        report("   nodal displacements", r.u, fin["nodal_disp"])
        nl = r.elem_layer > 0
        for L in np.unique(r.elem_layer):
            m = r.elem_layer == L
            print("     layer %d mean Mr: nartfem %.0f  MatFEA %.0f" % (L, r.elem_Mr[m].mean(), np.asarray(fin["elem_Mr"])[m].mean()))

    # ---- figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
        Mm = np.asarray(fin["elem_Mr"]); nlm = np.asarray(fin["elem_layer"]) > 1
        for scheme, mk in (("picard", "."), ("anderson", "x")):
            r = out[scheme]
            ax[0].loglog(Mm[nlm], r.elem_Mr[nlm], mk, ms=4, label="nartfem " + scheme)
        lim = [Mm[nlm].min() * 0.8, Mm[nlm].max() * 1.2]
        ax[0].plot(lim, lim, "k-", lw=0.8); ax[0].set_xlabel("MatFEA Mr (psi)"); ax[0].set_ylabel("nartfem Mr (psi)")
        ax[0].legend(); ax[0].set_title("converged moduli, nonlinear elements")
        ez_m = np.asarray(fin["elem_strain"])[:, 1]
        ax[1].plot(ez_m * 1e6, out["anderson"].elem_strain[:, 1] * 1e6, ".", ms=3)
        l2 = [ez_m.min() * 1e6, ez_m.max() * 1e6]; ax[1].plot(l2, l2, "k-", lw=0.8)
        ax[1].set_xlabel("MatFEA eps_z (microstrain)"); ax[1].set_ylabel("nartfem eps_z"); ax[1].set_title("vertical strain (Anderson)")
        cl = mesh.center[:, 0] <= 0.5 * model.radius_in
        zc = -mesh.center[cl, 1]
        o = np.argsort(zc)
        ax[2].plot(Mm[cl][o] / 1e3, zc[o], "k-", label="MatFEA")
        ax[2].plot(out["picard"].elem_Mr[cl][o] / 1e3, zc[o], "b--", label="nartfem picard")
        ax[2].plot(out["anderson"].elem_Mr[cl][o] / 1e3, zc[o], "r:", label="nartfem anderson")
        ax[2].set_ylim(1.3 * model.thickness_cum[-1], 0); ax[2].set_xlabel("Mr (ksi)"); ax[2].set_ylabel("depth (in)")
        ax[2].set_title("moduli near the axis"); ax[2].legend()
        fig.suptitle(model.name, fontsize=9); fig.tight_layout()
        png = os.path.splitext(dump_path)[0] + "_compare.png"
        fig.savefig(png, dpi=130); print("\nfigure written:", png)
    except Exception as e:                                        # matplotlib optional
        print("figure skipped:", e)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    main(*sys.argv[1:3])
