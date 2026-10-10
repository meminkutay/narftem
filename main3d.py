"""nartfem.solid3d - run a 3D model (all tires in the model) and plot cuts.
Press Run in PyCharm: CONFIG below is used.   python main3d.py <json> also works.
Install pypardiso (pip install pypardiso) for a ~10x faster solver; pyamg is the next best.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from nartfem.solid3d import load_model3d, run3d

CONFIG = dict(
    json3d="examples/FL_AC3_B5_SG4k_GG_dual3d.json",
    fine_in=None, z_scale=None, x_max_factor=None,    # override the JSON mesh settings (None = keep)
    backend=None,                                     # None = auto (pardiso > pyamg > splu > cg)
    max_iter=None,
    plots=True, show=True,
)


def main(cfg):
    import matplotlib
    if not cfg["show"]:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = load_model3d(cfg["json3d"])
    if cfg["fine_in"]: m.fine_in = cfg["fine_in"]
    if cfg["z_scale"]: m.z_scale = cfg["z_scale"]
    if cfg["x_max_factor"]: m.x_max_in = m.y_max_in = cfg["x_max_factor"] * m.radius_ref
    if cfg["backend"]: m.backend = cfg["backend"]
    if cfg["max_iter"]: m.max_iter = cfg["max_iter"]
    r = run3d(m)
    th = m.thickness_cum
    for t in m.tires:
        if t.x_in >= 0:
            print("under tire (%.2f, %.2f): eps_x/eps_y AC bottom %.0f / %.0f ue, eps_z SG top %.0f ue" % (
                t.x_in, t.y_in, r.fe_at("strain_x", t.x_in, t.y_in, th[0], "above") * 1e6,
                r.fe_at("strain_y", t.x_in, t.y_in, th[0], "above") * 1e6, r.fe_at("strain_z", t.x_in, t.y_in, th[-1], "below") * 1e6))
    print("between tires (0, 0): eps_x/eps_y AC bottom %.0f / %.0f ue, eps_z SG top %.0f ue" % (
        r.fe_at("strain_x", 0, 0, th[0], "above") * 1e6, r.fe_at("strain_y", 0, 0, th[0], "above") * 1e6,
        r.fe_at("strain_z", 0, 0, th[-1], "below") * 1e6))
    if cfg["plots"]:
        base = os.path.splitext(cfg["json3d"])[0]
        fig, ax = plt.subplots(2, 2, figsize=(14, 9))
        r.plot_cut("Mr", "xz", 0.0, ax=ax[0, 0]); r.plot_cut("strain_z", "xz", 0.0, ax=ax[0, 1])
        r.plot_cut("delsig" if np.any(r.elem_delsig > 0) else "stress_z", "xz", 0.0, ax=ax[1, 0])
        r.plot_cut("Mr", "xy", 0.5 * (th[0] + th[1]), ax=ax[1, 1])
        fig.tight_layout(); fig.savefig(base + "_cuts.png", dpi=120); print("written:", base + "_cuts.png")
        if cfg["show"]:
            plt.show()
    return r


if __name__ == "__main__":
    if len(sys.argv) > 1:
        CONFIG["json3d"] = sys.argv[1]
    main(CONFIG)
