"""nartfem strain app engine vs MatLEA (layered elastic theory) for the same loads on linear structures.

    python strain_app/verification/verify_vs_matlea.py

MatLEA is the Python port in _UPDAPSAC_MinersLaw (same algorithm and integration grid as f_MatLEA5.m).
Two MatLEA columns are printed:
    MatLEA    the code as it is (integration grid N1 = [20, 15, 10, 10] points between Bessel zeros)
    MatLEA*   the same code with a 16 x finer integration grid (converged reference)
The superposition (tensor rotation, peak search on the same x-y grid) is identical for both codes.
"""
import math, os, sys, time, types, warnings
import numpy as np
warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT, os.path.join(ROOT, "strain_app")]
import nartfem_strains as ns

def load_matlea(refine):
    for folder in ("_UPDAPSAC_MinersLaw", "legacy"):
        p = os.path.join(ROOT, folder, "MatLEA_MAIN_CODEMOD.py")
        if os.path.isfile(p):
            break
    src = open(p).read()
    old = "n1 = array([20,15,10,10])"
    assert old in src
    mod = types.ModuleType("MatLEA_x%d" % refine)
    exec(compile(src.replace(old, old + " * %d" % refine), p, "exec"), mod.__dict__)
    return mod.MatLEA

MATLEA = {1: load_matlea(1), 16: load_matlea(16)}

def matlea_superposed(MatLEA, TH, PR, E, circles, X, Y, z_t, z_sg, z_above):
    C = np.array(circles, float); top = np.concatenate([[0.0], np.cumsum(TH)]); n = len(E)
    pts = [(z_t, 1), (z_sg, n), (z_above, n - 1)]                       # (depth, layer number)
    Z = [[z for z, _ in pts], [z - top[l - 1] for z, l in pts], [l for _, l in pts]]
    XX, YY = np.meshgrid(X, Y, indexing="ij")
    out = {k: np.zeros(XX.shape) for k in ("exx", "eyy", "ezz", "ezza")}
    radii = np.unique(np.round(C[:, 2], 9)); rmax = 1.1 * math.hypot(X[-1] - X[0], Y[-1] - Y[0]); prof = {}
    for a in radii:
        p = float(C[np.isclose(C[:, 2], a), 3][0])
        RC = np.unique(np.concatenate([np.linspace(a * 0.001, 2 * a, 60), np.geomspace(2 * a, rmax, 150)]))
        o = MatLEA(Z, RC, p, a, np.array(E, float), np.array(PR, float), np.array(TH, float), None)
        r = np.asarray(o["r"]); z = np.asarray(o["z"])
        def pick(key, zz):
            m = np.isclose(z, zz); i = np.argsort(r[m]); return r[m][i], np.asarray(o[key])[m][i]
        prof[a] = (p, pick("strain_r", z_t), pick("strain_t", z_t), pick("strain_z", z_sg), pick("strain_z", z_above))
    for cx, cy, a, p in C:
        p0, er, et, ez, eza = prof[radii[np.argmin(abs(radii - a))]]
        dx, dy = XX - cx, YY - cy; R = np.hypot(dx, dy); th = np.arctan2(dy, dx); c, s = np.cos(th), np.sin(th); f = p / p0
        ER = np.interp(R, *er); ET = np.interp(R, *et)
        out["exx"] += f * (c * c * ER + s * s * ET); out["eyy"] += f * (s * s * ER + c * c * ET)
        out["ezz"] += f * np.interp(R, *ez); out["ezza"] += f * np.interp(R, *eza)
    return {k: v * 1e6 for k, v in out.items()}

def case(name, TH, PR, E, load):
    d = {"name": name, "load": load, "output": {"asphalt_bottom_layer": 1},
         "layers": [{"name": "L%d" % (i + 1), "type": "elastic", "thickness_in": (TH[i] if i < len(TH) else 0),
                     "E_psi": E[i], "nu": PR[i], "unit_weight_pcf": 140} for i in range(len(E))]}
    t = time.time(); r = ns.analyze(d, log=lambda *a: None); tn = time.time() - t
    X = np.array(r["profile"]["x_in"]); Y = np.array(r["profile"]["y_lines_in"]); H = sum(TH)
    args = (TH, PR, E, ns.build_load(load)["circles"], X, Y, TH[0] * 0.99999, H * 1.00001, H * 0.999)
    m1 = matlea_superposed(MATLEA[1], *args); m = matlea_superposed(MATLEA[16], *args)
    c = r["critical"]
    rows = [("eps_xx max, AC bottom", c["eps_xx_max"]["value"], m1["exx"].max(), m["exx"].max()),
            ("eps_yy max, AC bottom", c["eps_yy_max"]["value"], m1["eyy"].max(), m["eyy"].max()),
            ("eps_zz, top of subgrade", c["eps_zz_subgrade"]["value"], m1["ezz"].min(), m["ezz"].min()),
            ("eps_zz, 0.999 H (layer above)", c["eps_zz_above_interface"]["value"], m1["ezza"].min(), m["ezza"].min())]
    print("\n%s   [%d circles; nartfem %.1f s]" % (name, r["n_circles"], tn))
    print("  %-30s %9s | %9s %7s | %9s %7s" % ("microstrain", "nartfem", "MatLEA", "diff", "MatLEA*", "diff"))
    for lab, a_, b1, b2 in rows:
        print("  %-30s %9.1f | %9.1f %+6.1f%% | %9.1f %+6.1f%%" % (lab, a_, b1, 100 * (a_ - b1) / abs(b1), b2, 100 * (a_ - b2) / abs(b2)))

WB445 = {"type": "footprint", "rib_width_in": 1.42, "rib_lengths_in": [7.8, 7.8, 7.8, 7.8, 8.5, 8.2, 7.8, 7.3, 8.1], "rib_pressures_psi": [115, 115, 115, 125, 125, 125, 115, 115, 115]}
WB455 = {"type": "footprint", "rib_width_in": 1.34, "rib_lengths_in": [6.2, 7.6, 7.9, 8.3, 8.4, 8.3, 8.4, 10.3, 6.3], "rib_pressures_psi": [68, 103, 106, 113, 117, 113, 106, 103, 68]}
DUALFP = {"type": "footprint", "rib_width_in": 1.4, "rib_lengths_in": [7.8, 8.7, 9.3, 8.8, 8.0], "rib_pressures_psi": [100, 130, 125, 130, 100], "n_tires": 2, "tire_spacing_in": 11.9}
TH = [5, 8, 6, 6]; PR = [0.35, 0.40, 0.40, 0.40, 0.45]; E = [400000, 40000, 20000, 20000, 5000]     # structure of the FV_MatLEA_* scripts

if __name__ == "__main__":
    F = sum(p * l * 1.42 for p, l in zip(WB445["rib_pressures_psi"], WB445["rib_lengths_in"]))
    case("NGWB445 standard circle (101 in2)", TH, PR, E, {"type": "single", "pressure_psi": F / 101, "radius_in": math.sqrt(101 / math.pi)})
    case("Dual circles, 4.8 in radius, 13.5 in spacing, 120 psi", TH, PR, E, {"type": "dual", "pressure_psi": 120, "radius_in": 4.8, "spacing_in": 13.5})
    case("NGWB445 tire footprint", TH, PR, E, WB445)
    case("NGWB455 tire footprint", TH, PR, E, WB455)
    case("Dual-tire footprint", TH, PR, E, DUALFP)
    case("NGWB445 tire footprint, 3 in AC / 8 in base", [3, 8], [0.35, 0.4, 0.45], [400000, 30000, 5000], WB445)
    case("Single circle 4.8 in / 120 psi, 3 in AC / 5 in base", [3, 5], [0.25, 0.3, 0.35], [400000, 30000, 4000], {"type": "single", "pressure_psi": 120, "radius_in": 4.8})
