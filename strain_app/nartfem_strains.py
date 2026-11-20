"""nartfem_strains - critical pavement strains from a nartfem model JSON for single-circle,
dual-tire and measured tire-footprint (wide-base or dual) loads.

The model JSON is the nartfem.axi file (layers / domain / mesh / solver) with an extended
"load" block:

    "load": {"type": "single", "pressure_psi": 120, "radius_in": 4.8}          (type optional)
    "load": {"type": "dual",   "pressure_psi": 120, "radius_in": 4.8, "spacing_in": 13.5}
    "load": {"type": "footprint", "rib_width_in": 1.42,
             "rib_lengths_in": [...], "rib_pressures_psi": [...],
             "groove_width_in": 0.0, "n_tires": 1, "tire_spacing_in": 11.9}
    "load": {"type": "circles", "circles": [{"x_in": 0, "y_in": 0, "radius_in": 4, "pressure_psi": 100}, ...]}

x is transverse, y is the direction of travel.  A footprint rib (length x width rectangle at one
pressure) is replaced by a row of equal circles with the same area, hence the same pressure and
load (the "optimizer" rule of the MatLEA tire-footprint scripts).

Method
    one axisymmetric nartfem solution per distinct circle radius, then superposition of the
    strain tensors of all circles (tensor rotation to x-y), evaluated on an x-y grid at
        - the bottom of the asphalt  (horizontal strains eps_xx, eps_yy, major principal)
        - the top of the subgrade    (vertical strain eps_zz)
    Stress-dependent (universal / k-theta) layers:
        single circle      the nonlinear nartfem.axi solution
        identical circles  iterated superposition (nartfem.axi.multitire)
        footprint          the modulus-depth profile under an equivalent circle (same load and
                           contact area per tire; nonlinear solution) is frozen and the ribs are
                           superposed on that structure  - an approximation, reported in the log.

Optional blocks
    "output": {"asphalt_bottom_layer": 1}          1-based layer whose bottom is the tensile-strain plane
    "sweep":  {"AC.thickness_in": [3, 5, 7], "AC.E_psi": [200000, 400000]}     full factorial batch
              keys are "<layer name or 1-based number>.<field>" or "load.<field>"
"""
from __future__ import annotations
import copy
import itertools
import math
import re
import warnings
import numpy as np

from nartfem.axi import model_from_dict, run
from nartfem.axi.multitire import run_multitire

__version__ = "1.0"
MICRO = 1.0e6
MAX_SWEEP = 400


class InputError(ValueError):
    """A problem in the input file that the user can fix."""


# ============================================================================ load definition
def optimize_circles(length, width):
    """Number and radius of equal circles in a row that replace a length x width rectangle with
    the same area (port of f_optimize_circles.m)."""
    area = length * width

    def score(n):
        r = math.sqrt(area / (n * math.pi))
        return abs(2 * n * r - length) / length + abs(2 * r - width) / width
    best_n, best = 1, float("inf")
    for n in range(1, 10001):
        s = score(n)
        if s < best:
            best, best_n = s, n
        if n > 10 and s > best * 5:
            break
    for n in range(max(1, best_n - 100), best_n + 101):
        s = score(n)
        if s < best:
            best, best_n = s, n
    return best_n, math.sqrt(area / (best_n * math.pi))


def _num(d, key, what, default=None, positive=True):
    v = d.get(key, default)
    if v is None:
        raise InputError('%s: "%s" is missing.' % (what, key))
    try:
        v = float(v)
    except (TypeError, ValueError):
        raise InputError('%s: "%s" must be a number.' % (what, key))
    if positive and not v > 0:
        raise InputError('%s: "%s" must be greater than zero.' % (what, key))
    return v


def build_load(load):
    """-> dict(kind, circles (n,4) [x, y, radius, pressure], tires [(x, y, load_lb, area_in2)], notes)."""
    if not isinstance(load, dict):
        raise InputError('The "load" block is missing.')
    kind = str(load.get("type", "single")).lower().strip()
    notes = []
    if kind in ("single", "circle"):
        kind = "single"
        p = _num(load, "pressure_psi", "load")
        if load.get("radius_in") is None and load.get("load_lb") is not None:
            a = math.sqrt(_num(load, "load_lb", "load") / (p * math.pi))
        else:
            a = _num(load, "radius_in", "load")
        circles = [(0.0, 0.0, a, p)]
        tires = [(0.0, 0.0, p * math.pi * a * a, math.pi * a * a)]
    elif kind == "dual":
        p = _num(load, "pressure_psi", "load")
        if load.get("radius_in") is None and load.get("load_lb") is not None:
            a = math.sqrt(_num(load, "load_lb", "load") / (p * math.pi))
        else:
            a = _num(load, "radius_in", "load")
        s = _num(load, "spacing_in", "load (dual: center-to-center tire spacing)")
        if s < 2 * a:
            notes.append("WARNING: dual spacing %.2f in is smaller than the tire diameter %.2f in (the circles overlap)." % (s, 2 * a))
        circles = [(-s / 2, 0.0, a, p), (s / 2, 0.0, a, p)]
        tires = [(c[0], 0.0, p * math.pi * a * a, math.pi * a * a) for c in circles]
    elif kind == "footprint":
        w = _num(load, "rib_width_in", "load (footprint)")
        L = load.get("rib_lengths_in"); P = load.get("rib_pressures_psi")
        if not isinstance(L, list) or not isinstance(P, list) or not L or len(L) != len(P):
            raise InputError('load (footprint): "rib_lengths_in" and "rib_pressures_psi" must be lists of the same length.')
        try:
            L = [float(v) for v in L]; P = [float(v) for v in P]
        except (TypeError, ValueError):
            raise InputError("load (footprint): rib lengths and pressures must be numbers.")
        if min(L) <= 0 or min(P) <= 0:
            raise InputError("load (footprint): rib lengths and pressures must be greater than zero.")
        groove = _num(load, "groove_width_in", "load (footprint)", 0.0, positive=False)
        n_tires = int(load.get("n_tires", 1))
        if n_tires not in (1, 2):
            raise InputError('load (footprint): "n_tires" must be 1 (wide-base) or 2 (dual).')
        nr = [optimize_circles(Lk, w) for Lk in L]
        nrib = len(L); mid = (nrib - 1) // 2                       # central rib at x = 0 (rib 5 of 9, 3 of 5)
        xc = [0.0] * nrib
        for k in range(mid + 1, nrib):
            xc[k] = xc[k - 1] + nr[k - 1][1] + nr[k][1] + groove
        for k in range(mid - 1, -1, -1):
            xc[k] = xc[k + 1] - nr[k + 1][1] - nr[k][1] - groove
        one = []
        for k in range(nrib):
            n, r = nr[k]
            for y in np.linspace(-(n - 1) * r, (n - 1) * r, n):
                one.append((xc[k], float(y), r, P[k]))
        force = sum(Pk * Lk * w for Pk, Lk in zip(P, L)); area = sum(Lk * w for Lk in L)
        if n_tires == 1:
            circles = one; tires = [(0.0, 0.0, force, area)]
        else:
            s = _num(load, "tire_spacing_in", "load (footprint, n_tires = 2: center-to-center tire spacing)")
            circles = [(c[0] - s / 2, c[1], c[2], c[3]) for c in one] + [(c[0] + s / 2, c[1], c[2], c[3]) for c in one]
            tires = [(-s / 2, 0.0, force, area), (s / 2, 0.0, force, area)]
        notes.append("footprint: %d ribs per tire -> %d circles per tire (circles per rib: %s; radii %.3f-%.3f in)"
                     % (nrib, len(one), " ".join(str(n) for n, _ in nr), min(r for _, r in nr), max(r for _, r in nr)))
        notes.append("footprint: load per tire %.0f lb on %.1f in^2 (mean pressure %.1f psi, equivalent circle radius %.3f in)"
                     % (force, area, force / area, math.sqrt(area / math.pi)))
    elif kind == "circles":
        cl = load.get("circles")
        if not isinstance(cl, list) or not cl:
            raise InputError('load (circles): "circles" must be a list of {"x_in", "y_in", "radius_in", "pressure_psi"}.')
        circles = []
        for i, c in enumerate(cl):
            what = "load (circle %d)" % (i + 1)
            circles.append((_num(c, "x_in", what, positive=False), _num(c, "y_in", what, 0.0, positive=False),
                            _num(c, "radius_in", what), _num(c, "pressure_psi", what)))
        tires = [(c[0], c[1], c[3] * math.pi * c[2] ** 2, math.pi * c[2] ** 2) for c in circles]
    else:
        raise InputError('load: unknown "type" "%s" (use single, dual, footprint or circles).' % kind)
    C = np.array(circles, float)
    return dict(kind=kind, circles=C, tires=tires, notes=notes,
                total_load_lb=float(np.sum(C[:, 3] * np.pi * C[:, 2] ** 2)))


# ============================================================================ model
_LAYER_FIELDS = ("name", "type", "thickness_in", "E_psi", "nu", "unit_weight_pcf", "k", "K0", "residual_psi",
                 "Mr_min_psi", "E_table", "Er_over_Ez", "nu_hor", "geogrids")


def build_model(d, radius, pressure):
    layers = d.get("layers")
    if not isinstance(layers, list) or len(layers) < 2:
        raise InputError('"layers" must list at least two layers (the last one is the subgrade).')
    for i, L in enumerate(layers):
        what = "layer %d" % (i + 1)
        if not isinstance(L, dict):
            raise InputError(what + " must be a { ... } block.")
        for key in L:
            if key not in _LAYER_FIELDS:
                raise InputError('%s: unknown entry "%s". Allowed: %s.' % (what, key, ", ".join(_LAYER_FIELDS)))
        t = L.get("type")
        if t not in ("elastic", "elastic_depth", "universal", "ktheta"):
            raise InputError('%s: "type" must be elastic, elastic_depth, universal or ktheta.' % what)
        if i < len(layers) - 1:
            _num(L, "thickness_in", what)
        if t == "elastic":
            _num(L, "E_psi", what)
        if t in ("universal", "ktheta") and len(L.get("k") or []) < (3 if t == "universal" else 2):
            raise InputError('%s: "k" needs %s.' % (what, "[k1, k2, k3]" if t == "universal" else "[k1_psi, k2]"))
        if t == "elastic_depth" and not L.get("E_table"):
            raise InputError('%s: "E_table" [[depth_in, E_psi], ...] is missing.' % what)
        nu = _num(L, "nu", what, 0.35)
        if not 0 < nu < 0.5:
            raise InputError("%s: Poisson's ratio must be between 0 and 0.5." % what)
    if (d.get("mesh") or {}).get("import_file"):
        raise InputError('mesh: "import_file" is not supported here; set it to null.')
    dd = dict(d); dd["load"] = {"pressure_psi": float(pressure), "radius_in": float(radius)}
    try:
        m = model_from_dict(dd)
    except TypeError as e:
        raise InputError("Could not read the model: %s" % e)
    m.verbose = False
    return m


def asphalt_bottom_layer(d, model):
    """1-based number of the layer whose bottom is the tensile-strain plane."""
    out = d.get("output") or {}
    n = out.get("asphalt_bottom_layer")
    if n is not None:
        n = int(n)
        if not 1 <= n <= model.n_layers - 1:
            raise InputError('output: "asphalt_bottom_layer" must be between 1 and %d.' % (model.n_layers - 1))
        return n
    n = 0
    for L in model.layers[:-1]:
        if re.search(r"(^|[^a-z])(ac|hma|wma|asphalt|sma)", L.name.lower()):
            n += 1
        else:
            break
    return max(n, 1)


# ============================================================================ radial profiles
class Profile:
    """Radial strain profiles of one single-circle solution at the two critical depths.

    The strains of the 4-node elements are sampled where they are most accurate: at the radial
    element centers (eps_r and eps_z are constant along r / z inside an element).  The vertical
    strain at the subgrade interface is extrapolated (quadratic in depth) from the first three
    element rows on each side of the interface, because the value inside the first element is
    the strain at its mid-height, not at the interface."""

    def __init__(self, res, model, z_t, z_c, rmax):
        mesh = res.mesh
        rc = np.unique(np.round(mesh.center[:, 0], 9))
        rc = rc[rc <= max(rmax, rc[min(3, rc.size - 1)]) * 1.05]
        n = rc.size
        zrows = np.unique(np.round(-mesh.center[:, 1], 9))
        er = res.fe_at("strain_r", rc, np.full(n, z_t), side="above")
        et = res.fe_at("strain_t", rc, np.full(n, z_t), side="above")
        et0 = float(res.fe_at("strain_t", np.array([0.0]), np.array([z_t]), side="above"))
        self.r = np.concatenate([[0.0], rc])
        self.er = np.concatenate([[et0], er]); self.et = np.concatenate([[et0], et])   # eps_r = eps_t on the axis

        def ez_at_interface(rows, elastic):
            vals = [res.fe_at("strain_z", rc, np.full(n, z), side="above") for z in rows[:3]]
            if not elastic or len(rows) < 3:
                return vals[0], False
            z1, z2, z3 = rows[:3]                                    # Lagrange quadratic extrapolation to z_c
            w1 = (z_c - z2) * (z_c - z3) / ((z1 - z2) * (z1 - z3))
            w2 = (z_c - z1) * (z_c - z3) / ((z2 - z1) * (z2 - z3))
            w3 = (z_c - z1) * (z_c - z2) / ((z3 - z1) * (z3 - z2))
            return w1 * vals[0] + w2 * vals[1] + w3 * vals[2], True
        below = zrows[zrows > z_c]
        z_prev = float(model.thickness_cum[-2]) if model.n_layers > 2 else 0.0
        above = zrows[(zrows < z_c) & (zrows > z_prev)][::-1]
        ez_sg, self.extrapolated = ez_at_interface(below, model.layers[-1].type == "elastic")
        ez_ab, _ = ez_at_interface(above, model.layers[-2].type == "elastic")
        self.ez_sg = np.concatenate([[ez_sg[0]], ez_sg]); self.ez_above = np.concatenate([[ez_ab[0]], ez_ab])

    def at(self, R):
        f = lambda v: np.interp(R, self.r, v)
        return f(self.er), f(self.et), f(self.ez_sg), f(self.ez_above)


def _frozen_dict(d, model, res, r_avg):
    """Copy of the input with every stress-dependent layer replaced by an elastic layer whose
    modulus varies with depth: the area-weighted mean modulus of ``res`` within r <= r_avg."""
    mesh = res.mesh
    xy = mesh.nodes[mesh.conn]
    area = 0.5 * np.abs(np.sum(xy[:, :, 0] * np.roll(xy[:, :, 1], -1, axis=1) -
                               np.roll(xy[:, :, 0], -1, axis=1) * xy[:, :, 1], axis=1))
    dd = copy.deepcopy(d)
    summary = []
    for i, L in enumerate(model.layers):
        if not L.is_nonlinear:
            continue
        m = (mesh.layer == i) & (mesh.center[:, 0] <= r_avg)
        depth = np.round(-mesh.center[m, 1], 9)
        tab = []
        for z in np.unique(depth):
            s = depth == z
            w = area[m][s] * mesh.center[m, 0][s]                    # ring volume weight
            tab.append([float(z), float(np.sum(res.elem_Mr[m][s] * w) / np.sum(w))])
        if len(tab) == 1:
            tab.append([tab[0][0] + 1.0, tab[0][1]])
        Ld = dd["layers"][i]
        new = {k: Ld[k] for k in ("name", "thickness_in", "nu", "unit_weight_pcf") if k in Ld}
        new["type"] = "elastic_depth"; new["E_table"] = tab
        dd["layers"][i] = new
        E = np.array(tab)[:, 1]
        summary.append("%s: %.0f psi at the top to %.0f psi at the bottom (mean %.0f psi)" % (L.name, E[0], E[-1], E.mean()))
    return dd, summary


# ============================================================================ one analysis
def analyze(d, log=print):
    """Run one model dict; returns a JSON-serializable result dict."""
    if not isinstance(d, dict):
        raise InputError("The input must be a JSON object { ... }.")
    load = build_load(d.get("load"))
    C = load["circles"]
    for n in load["notes"]:
        log(n)
    model0 = build_model(d, C[0, 2], C[0, 3])
    nonlinear = any(L.is_nonlinear for L in model0.layers)
    n_ac = asphalt_bottom_layer(d, model0)
    thc = model0.thickness_cum
    z_t, z_c = float(thc[n_ac - 1]), float(thc[-1])

    # ---- evaluation grid
    rmax_c = float(C[:, 2].max())
    x_lo, x_hi = float(C[:, 0].min()), float(C[:, 0].max())
    y_lo, y_hi = float(C[:, 1].min()), float(C[:, 1].max())
    margin = max(3 * rmax_c, 1.5 * z_c, 12.0)
    X = np.unique(np.round(np.concatenate([np.linspace(x_lo - margin, x_hi + margin, 401), np.unique(C[:, 0]), [0.5 * (x_lo + x_hi)]]), 6))
    y_mid = 0.5 * (y_lo + y_hi)
    sym_y = np.allclose(np.sort(C[:, 1] - y_mid), np.sort(-(C[:, 1] - y_mid)), atol=1e-9)
    half = 0.5 * (y_hi - y_lo) + rmax_c
    Y = y_mid + (np.linspace(0, half, 9) if sym_y else np.linspace(-half, half, 17))
    reach = 1.05 * math.hypot(X[-1] - X[0], Y[-1] - Y[0])

    # ---- domain width: the default 40 x radius is too small for the small circles of a footprint
    r_group = math.sqrt(float(np.sum(C[:, 2] ** 2)))                 # radius of one circle with the total contact area
    width_user = (d.get("domain") or {}).get("width_in")
    width = float(width_user) if width_user else max(40 * rmax_c, 40 * r_group, 4 * reach)
    if width < 1.5 * reach:
        raise InputError("domain.width_in = %.0f in is too small for this load (use at least %.0f in, or null)." % (width, 2 * reach))

    def solve(dd, radius, pressure, tag):
        m = build_model(dd, radius, pressure)
        m.width_in = width
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            res = run(m)
        for w in wl:
            log("WARNING (%s): %s" % (tag, w.message))
        return m, res

    info = dict(method="", converged=True, iterations=0, moduli=[])
    radii = np.unique(np.round(C[:, 2], 9))
    profiles = {}                                                    # radius -> (Profile, reference pressure)
    identical = radii.size == 1 and np.unique(np.round(C[:, 3], 9)).size == 1

    def record(res, m):
        info["converged"] = bool(res.converged); info["iterations"] = int(res.iterations)
        info["moduli"] = [dict(layer=r["layer"], name=r["name"], mean_all=float(r["mean_all"]),
                               mean_near=float(r["mean_near"])) for r in res.layer_summary(log=None)]
        if nonlinear:
            log("nonlinear solution: %s after %d iterations (error %.2f%%)" % (
                "converged" if res.converged else "NOT CONVERGED", res.iterations, 100 * res.conv_error))

    if len(C) == 1:
        info["method"] = "single circle, axisymmetric nartfem solution" + (" (nonlinear)" if nonlinear else "")
        m, res = solve(d, C[0, 2], C[0, 3], "single")
        record(res, m)
        profiles[radii[0]] = (Profile(res, m, z_t, z_c, reach), C[0, 3])
    elif nonlinear and identical:
        info["method"] = "%d identical circles, iterated superposition with stress-dependent moduli" % len(C)
        m = build_model(d, C[0, 2], C[0, 3]); m.width_in = width
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            res, _ = run_multitire(m, [(c[0], c[1]) for c in C], log=lambda *a, **k: None)
        for w in wl:
            log("WARNING: %s" % w.message)
        record(res, m)
        profiles[radii[0]] = (Profile(res, m, z_t, z_c, reach), C[0, 3])
    else:
        dd = d
        if nonlinear:
            if load["kind"] != "footprint":
                raise InputError("Circles of different size or pressure together with stress-dependent (universal / ktheta) "
                                 "layers are only supported for the footprint load type.")
            T = load["tires"]
            r_eq = math.sqrt(T[0][3] / math.pi); p_eq = T[0][2] / T[0][3]
            log("stress-dependent layers: moduli from the equivalent circle of each tire (radius %.3f in, %.1f psi), "
                "then frozen for the rib superposition (approximation)" % (r_eq, p_eq))
            m = build_model(d, r_eq, p_eq); m.width_in = width
            with warnings.catch_warnings(record=True) as wl:
                warnings.simplefilter("always")
                if len(T) == 1:
                    res = run(m)
                else:
                    res, _ = run_multitire(m, [(t[0], t[1]) for t in T], log=lambda *a, **k: None)
            for w in wl:
                log("WARNING: %s" % w.message)
            record(res, m)
            dd, summ = _frozen_dict(d, m, res, r_eq)
            for s in summ:
                log("  frozen modulus, " + s)
            info["method"] = "%d circles, superposition on the modulus profile of the equivalent-circle nonlinear solution" % len(C)
        else:
            info["method"] = "%d circles (%d distinct radii), linear superposition" % (len(C), radii.size)
        for j, a in enumerate(radii):
            p_ref = float(C[np.isclose(C[:, 2], a), 3][0])
            log("solving circle radius %.4f in (%d of %d)" % (a, j + 1, radii.size))
            m, res = solve(dd, a, p_ref, "radius %.3f" % a)
            profiles[a] = (Profile(res, m, z_t, z_c, reach), p_ref)
            if not nonlinear and j == 0:
                record(res, m)

    # ---- superposition
    XX, YY = np.meshgrid(X, Y, indexing="ij")
    exx = np.zeros(XX.shape); eyy = np.zeros(XX.shape); exy = np.zeros(XX.shape)
    ezz = np.zeros(XX.shape); ezz_ab = np.zeros(XX.shape)
    for (cx, cy, a, p) in C:
        prof, p_ref = profiles[radii[np.argmin(np.abs(radii - a))]]
        dx, dy = XX - cx, YY - cy
        R = np.hypot(dx, dy); th = np.arctan2(dy, dx)
        er, et, ez, eza = prof.at(R)
        f = p / p_ref
        c, s = np.cos(th), np.sin(th)
        exx += f * (c * c * er + s * s * et); eyy += f * (s * s * er + c * c * et)
        exy += f * (c * s * (er - et)); ezz += f * ez; ezz_ab += f * eza
    e1 = 0.5 * (exx + eyy) + np.sqrt((0.5 * (exx - eyy)) ** 2 + exy ** 2)

    def peak(A, mode):
        k = np.unravel_index(np.argmax(A) if mode == "max" else np.argmin(A), A.shape)
        return dict(value=float(A[k] * MICRO), x_in=float(X[k[0]]), y_in=float(Y[k[1]])), k
    p_xx, k_xx = peak(exx, "max"); p_yy, k_yy = peak(eyy, "max"); p_e1, k_e1 = peak(e1, "max")
    p_zz, k_zz = peak(ezz, "min"); p_zza, _ = peak(ezz_ab, "min")
    jt = k_e1[1]; jc = k_zz[1]
    result = dict(
        name=str(d.get("name", "")), load_type=load["kind"], n_circles=int(len(C)),
        total_load_lb=load["total_load_lb"], circles=np.round(C, 5).tolist(),
        asphalt_bottom_layer=n_ac, z_tensile_in=z_t, z_subgrade_in=z_c,
        layers=[dict(name=L.name, type=L.type, thickness_in=(float(L.thickness_in) if i < model0.n_layers - 1 else None))
                for i, L in enumerate(model0.layers)],
        info=info, nonlinear=bool(nonlinear),
        critical=dict(eps_xx_max=p_xx, eps_yy_max=p_yy, eps_h_principal_max=p_e1,
                      eps_h_max=(p_xx if p_xx["value"] >= p_yy["value"] else p_yy),
                      eps_h_direction=("xx (transverse)" if p_xx["value"] >= p_yy["value"] else "yy (longitudinal)"),
                      eps_zz_subgrade=p_zz, eps_zz_above_interface=p_zza),
        profile=dict(x_in=np.round(X, 4).tolist(), y_lines_in=np.round(Y, 4).tolist(), y_tensile_in=float(Y[jt]), y_subgrade_in=float(Y[jc]),
                     eps_xx=np.round(exx[:, jt] * MICRO, 4).tolist(), eps_yy=np.round(eyy[:, jt] * MICRO, 4).tolist(),
                     eps_zz=np.round(ezz[:, jc] * MICRO, 4).tolist()))
    log("asphalt bottom (z = %.3f in): max eps_xx %+.1f, max eps_yy %+.1f microstrain" % (z_t, p_xx["value"], p_yy["value"]))
    log("subgrade top   (z = %.3f in): eps_zz %+.1f microstrain" % (z_c, p_zz["value"]))
    return result


# ============================================================================ sweep (batch)
def _set_param(d, key, value):
    if "." not in key:
        raise InputError('sweep: "%s" must look like "AC.thickness_in" or "load.pressure_psi".' % key)
    target, field = key.rsplit(".", 1)
    target = target.strip(); field = field.strip()
    if target.lower() == "load":
        if field not in d.get("load", {}):
            raise InputError('sweep: the load block has no entry "%s".' % field)
        d["load"][field] = value
        return
    layers = d.get("layers") or []
    idx = None
    if target.isdigit() and 1 <= int(target) <= len(layers):
        idx = int(target) - 1
    else:
        hits = [i for i, L in enumerate(layers) if str(L.get("name", "")).lower() == target.lower()]
        if len(hits) == 1:
            idx = hits[0]
        elif len(hits) > 1:
            raise InputError('sweep: more than one layer is named "%s"; use the layer number instead.' % target)
    if idx is None:
        raise InputError('sweep: no layer named "%s" (use a layer name or its number, 1 = top).' % target)
    if field not in _LAYER_FIELDS:
        raise InputError('sweep: unknown layer entry "%s".' % field)
    layers[idx][field] = value


def expand_sweep(d):
    """-> list of (label dict, model dict).  Without a sweep block: one case."""
    sw = d.get("sweep")
    base = {k: v for k, v in d.items() if k != "sweep"}
    if not sw:
        return [({}, base)]
    if not isinstance(sw, dict):
        raise InputError('"sweep" must be a block like {"AC.thickness_in": [3, 5, 7]}.')
    keys = list(sw)
    for k in keys:
        if not isinstance(sw[k], list) or not sw[k]:
            raise InputError('sweep: "%s" must be a non-empty list of values.' % k)
    n = int(np.prod([len(sw[k]) for k in keys]))
    if n > MAX_SWEEP:
        raise InputError("sweep: %d combinations requested; the limit is %d." % (n, MAX_SWEEP))
    cases = []
    for combo in itertools.product(*[sw[k] for k in keys]):
        dd = copy.deepcopy(base)
        for k, v in zip(keys, combo):
            _set_param(dd, k, v)
        cases.append((dict(zip(keys, combo)), dd))
    return cases


def summary_row(label, r):
    c = r["critical"]
    row = dict(label)
    row.update({"eps_xx_max": c["eps_xx_max"]["value"], "eps_yy_max": c["eps_yy_max"]["value"],
                "eps_h_max": c["eps_h_max"]["value"], "eps_zz_subgrade": c["eps_zz_subgrade"]["value"],
                "eps_zz_above_interface": c["eps_zz_above_interface"]["value"],
                "converged": r["info"]["converged"]})
    return row


def run_all(d, log=print, progress=None, cancelled=None):
    """Run the input (single case or sweep).  Returns dict(cases=[...], rows=[...], sweep_keys=[...])."""
    cases = expand_sweep(d)
    out = dict(cases=[], rows=[], sweep_keys=list(cases[0][0]) if cases and cases[0][0] else [])
    for i, (label, dd) in enumerate(cases):
        if cancelled and cancelled():
            log("stopped by the user."); break
        if len(cases) > 1:
            log("")
            log("=== case %d of %d: %s" % (i + 1, len(cases), ", ".join("%s = %s" % kv for kv in label.items())))
        r = analyze(dd, log=log)
        r["label"] = label
        out["cases"].append(r); out["rows"].append(summary_row(label, r))
        if progress:
            progress(i + 1, len(cases))
    return out
