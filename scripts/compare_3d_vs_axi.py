"""3D dual-tire solution vs axisymmetric single tire + superposition (the MEAPA approach).

Both are evaluated exactly inside the elements (fe_at).  The superposition rotates the
single-tire strain tensor of each tire to x-y and sums (tensor shear, principal strains of the
sum) - the corrected ff_superpose_strains logic.  Profiles along x (transverse, through both
tires, y = 0) at the AC bottom and the subgrade top, and the base modulus at mid-base.

Run from PyCharm (CONFIG) or:  python scripts/compare_3d_vs_axi.py examples/FL_AC3_B5_SG4k_GG_dual3d.json
Writes <json>_vs_axi.png and prints a table.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from nartfem.solid3d import load_model3d, run3d
from nartfem.axi import Model, run
from nartfem.axi.multitire import run_multitire, superposed_strain

# ---------------------------------------------------------------- CONFIG (PyCharm "Run")
CONFIG = dict(
    json3d="examples/FL_AC3_B5_SG4k_GG_dual3d.json",
    fine_in=None, z_scale=None, x_max_factor=None,   # override the JSON mesh settings (None = keep)
    max_iter=120,
    show=False,
)
# ------------------------------------------------------------------------------------


def axi_single(model3d, tire):
    """Axisymmetric model of one tire of the 3D model (same layers, solver settings)."""
    m = model3d.axi_model()
    m.pressure_psi = tire.pressure_psi; m.radius_in = tire.radius_in; m.verbose = False
    m.max_iter = max(m.max_iter, 200); m.max_error_pct = 2.0        # the axi engine's own criterion
    return run(m)


def superpose(res_axi, tires, x, y, depth, side):
    """Strain tensor (xx, yy, zz, xy, xz, yz) at points (x, y, depth) from the single-tire field."""
    x = np.atleast_1d(x); y = np.atleast_1d(y)
    exx = np.zeros(x.size); eyy = np.zeros(x.size); ezz = np.zeros(x.size); exy = np.zeros(x.size)
    for t in tires:
        dx, dy = x - t.x_in, y - t.y_in
        r = np.hypot(dx, dy); th = np.arctan2(dy, dx)
        er = np.array([res_axi.fe_at("strain_r", ri, depth, side=side) for ri in r])
        et = np.array([res_axi.fe_at("strain_t", ri, depth, side=side) for ri in r])
        ez = np.array([res_axi.fe_at("strain_z", ri, depth, side=side) for ri in r])
        c, s = np.cos(th), np.sin(th)
        exx += c ** 2 * er + s ** 2 * et; eyy += s ** 2 * er + c ** 2 * et; exy += c * s * (er - et); ezz += ez
    return exx, eyy, ezz, exy


def main(cfg):
    import matplotlib
    if not cfg["show"]:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m3 = load_model3d(cfg["json3d"])
    if cfg["fine_in"]: m3.fine_in = cfg["fine_in"]
    if cfg["z_scale"]: m3.z_scale = cfg["z_scale"]
    if cfg["x_max_factor"]: m3.x_max_in = m3.y_max_in = cfg["x_max_factor"] * m3.radius_ref
    m3.max_iter = cfg["max_iter"]
    t0 = time.time(); r3 = run3d(m3); t3 = time.time() - t0
    t0 = time.time(); ra = axi_single(m3, m3.tires[0]); ta = time.time() - t0
    tires_xy = [(t.x_in, t.y_in) for t in m3.tires]
    ma = m3.axi_model(); ma.pressure_psi = m3.tires[0].pressure_psi; ma.radius_in = m3.tires[0].radius_in; ma.verbose = False; ma.max_iter = 200; ma.max_error_pct = 2.0
    t0 = time.time(); rm, _ = run_multitire(ma, tires_xy, n_phi=8); tm = time.time() - t0
    th = m3.thickness_cum
    zac, zsg, zmid = th[0], th[-1], 0.5 * (th[0] + th[1])
    xmax = max(abs(t.x_in) for t in m3.tires) + 3 * m3.radius_ref
    xx = np.linspace(0, xmax, 61); yy = np.zeros_like(xx)

    # AC bottom: horizontal strains (xx = transverse, yy = longitudinal), max principal in-plane
    e3 = {k: np.array([r3.fe_at("strain_" + k, xi, 0.0, zac, side="above") for xi in xx]) for k in ("x", "y", "xy")}
    sx, sy, _, sxy = superpose(ra, m3.tires, xx, yy, zac, "above")
    mx, my, _, mxy = superposed_strain(rm, tires_xy, xx, yy, zac, "above")
    def pmax(a, b, g):      # max principal of the in-plane tensor (engineering shear g)
        return 0.5 * (a + b) + np.sqrt((0.5 * (a - b)) ** 2 + (0.5 * g) ** 2)
    p3 = pmax(e3["x"], e3["y"], e3["xy"]); ps = pmax(sx, sy, 2 * sxy); pm = pmax(mx, my, 2 * mxy)
    # subgrade top and mid-base vertical strains
    ez3 = np.array([r3.fe_at("strain_z", xi, 0.0, zsg, side="below") for xi in xx]); _, _, ezs, _ = superpose(ra, m3.tires, xx, yy, zsg, "below")
    _, _, ezm, _ = superposed_strain(rm, tires_xy, xx, yy, zsg, "below")
    ezb3 = np.array([r3.fe_at("strain_z", xi, 0.0, zmid) for xi in xx]); _, _, ezbs, _ = superpose(ra, m3.tires, xx, yy, zmid, "above")
    _, _, ezbm, _ = superposed_strain(rm, tires_xy, xx, yy, zmid, "above")
    # base modulus at mid-base along x: 3D element values vs the single-tire field around each tire
    E3 = np.array([r3.elem_Mr[r3.mesh.elem_index(*r3.mesh.locate(xi, 0.0, zmid))] for xi in xx])
    Ea_tire = np.array([ra.E_centerline(2, zmid) if False else np.nan for _ in xx])
    Ea = []
    for xi in xx:                        # superposition uses the single-tire moduli: take the nearest tire's field
        rmin = min(abs(xi - t.x_in) for t in m3.tires)
        e = np.argmin(np.hypot(ra.mesh.center[:, 0] - rmin, -ra.mesh.center[:, 1] - zmid) + 1e9 * (ra.mesh.layer != 1))
        Ea.append(ra.elem_Mr[e])
    Ea = np.array(Ea)
    Em = []
    for xi in xx:
        rmin = min(abs(xi - t.x_in) for t in m3.tires)
        e = np.argmin(np.hypot(rm.mesh.center[:, 0] - rmin, -rm.mesh.center[:, 1] - zmid) + 1e9 * (rm.mesh.layer != 1))
        Em.append(rm.elem_Mr[e])
    Em = np.array(Em)

    # ---- table at the two critical points: under a tire center and between the tires
    xt = abs(m3.tires[0].x_in)
    def at(arr, x0): return np.interp(x0, xx, arr)
    print("\n%s\n3D: %d elements, %d iterations (conv=%s), %.0f s | axi: %d it, %.0f s | iterated superposition: %d it, %.0f s"
          % (m3.name, r3.mesh.n_elements, r3.iterations, r3.converged, t3, ra.iterations, ta, rm.iterations, tm))
    print("%-44s %10s %12s %7s %12s %7s" % ("", "3D dual", "axi+superp", "diff", "axi+iter.sp", "diff"))
    for lab, a3, asup, am in [("AC bottom max principal strain, under tire", at(p3, xt) * 1e6, at(ps, xt) * 1e6, at(pm, xt) * 1e6),
                              ("AC bottom max principal strain, between tires", at(p3, 0) * 1e6, at(ps, 0) * 1e6, at(pm, 0) * 1e6),
                              ("AC bottom eps_xx (transverse), between tires", at(e3["x"], 0) * 1e6, at(sx, 0) * 1e6, at(mx, 0) * 1e6),
                              ("AC bottom eps_yy (longitudinal), under tire", at(e3["y"], xt) * 1e6, at(sy, xt) * 1e6, at(my, xt) * 1e6),
                              ("subgrade top eps_z, under tire", at(ez3, xt) * 1e6, at(ezs, xt) * 1e6, at(ezm, xt) * 1e6),
                              ("subgrade top eps_z, between tires", at(ez3, 0) * 1e6, at(ezs, 0) * 1e6, at(ezm, 0) * 1e6),
                              ("mid-base eps_z, under tire", at(ezb3, xt) * 1e6, at(ezbs, xt) * 1e6, at(ezbm, xt) * 1e6),
                              ("mid-base eps_z, between tires", at(ezb3, 0) * 1e6, at(ezbs, 0) * 1e6, at(ezbm, 0) * 1e6),
                              ("mid-base modulus, under tire (psi)", at(E3, xt), at(Ea, xt), at(Em, xt)),
                              ("mid-base modulus, between tires (psi)", at(E3, 0), at(Ea, 0), at(Em, 0))]:
        print("%-44s %10.1f %12.1f %6.1f%% %12.1f %6.1f%%" % (lab, a3, asup, 100 * (asup / a3 - 1), am, 100 * (am / a3 - 1)))

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    for a, (y3, ysup, ym, lab) in zip(ax.ravel(), [(p3 * 1e6, ps * 1e6, pm * 1e6, "AC bottom: max principal (tensile) strain (microstrain)"),
                                                    (ez3 * 1e6, ezs * 1e6, ezm * 1e6, "subgrade top: eps_z (microstrain)"),
                                                    (ezb3 * 1e6, ezbs * 1e6, ezbm * 1e6, "mid-base: eps_z (microstrain)"),
                                                    (E3 / 1e3, Ea / 1e3, Em / 1e3, "mid-base modulus (ksi)")]):
        a.plot(xx, y3, "-", lw=2, label="3D, all tires in the model")
        a.plot(xx, ysup, "--", lw=1.5, label="axisymmetric + superposition (MEAPA)")
        a.plot(xx, ym, ":", lw=2, label="axisymmetric + iterated superposition")
        for t in m3.tires:
            if t.x_in >= 0:
                a.axvspan(t.x_in - t.radius_in, t.x_in + t.radius_in, color="0.9")
        a.set_xlabel("x (in), transverse through the dual tires (y = 0)"); a.set_ylabel(lab); a.grid(True, alpha=0.3)
    ax[0, 0].legend()
    fig.suptitle(m3.name + "\n(shaded: tire contact; symmetric about x = 0)", fontsize=10); fig.tight_layout()
    out = os.path.splitext(cfg["json3d"])[0] + "_vs_axi.png"; fig.savefig(out, dpi=120); print("written:", out)
    if cfg["show"]:
        plt.show()
    return r3, ra


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    if len(sys.argv) > 1:
        CONFIG["json3d"] = sys.argv[1]
    main(CONFIG)
