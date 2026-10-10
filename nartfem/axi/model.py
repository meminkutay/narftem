"""Model definition for nartfem.axi (the Python counterpart of the MATLAB ``Model`` struct
built in ff_run_MatFEA.m / ExampleNonlin_3Lyr.m).

Units: inches, psi, lb, pcf (unit weight in the input; converted to pci internally),
       degrees, kN/m for the geogrid sheet stiffness.  Depth is positive downward in the
       input; internally the FE uses z <= 0 (MatFEA convention).

A model can be built in Python (dataclasses below) or read from a JSON file::

    {
      "name": "FL 3in AC / 5in base / 4000 psi subgrade, geogrid at base bottom",
      "load":   {"pressure_psi": 120, "radius_in": 4.887},
      "layers": [
        {"name": "AC",   "type": "elastic", "thickness_in": 3, "E_psi": 450000, "nu": 0.35,
         "unit_weight_pcf": 145},
        {"name": "Base", "type": "universal", "thickness_in": 5, "k": [1500, 0.85, -0.15],
         "nu": 0.40, "unit_weight_pcf": 135, "K0": 0.5, "residual_psi": 3.0, "Mr_min_psi": 1000,
         "geogrids": [{"l_in": 0.5, "delta_in": 6, "M_kNm": 428, "vg": 0.35, "phi_deg": 51.3,
                       "c_psi": 2.93, "eps0": 0.149, "rho": 72.4, "beta": 0.247,
                       "m": 1.72, "n": -2.16, "N": 1.0e6}]},
        {"name": "Subgrade", "type": "elastic", "E_psi": 4000, "nu": 0.45,
         "unit_weight_pcf": 120}
      ],
      "domain": {"width_in": null, "depth_in": null},
      "mesh":   {"scale": 1.3, "import_file": null},
      "solver": {"gauss_points": 16, "max_iter": 20, "reliability_pct": 95,
                 "max_error_pct": 2.0, "atm_pressure_psi": 14.706}
    }

Layer types
    "elastic"        constant modulus E_psi
    "elastic_depth"  modulus from a table  "E_table": [[depth_in, E_psi], ...]  (MatFEA 'Elastic:MrFunc')
    "universal"      NCHRP 1-37A universal model, k = [k1, k2, k3]  (k1 dimensionless; Mr = k1*Pa*(..))
    "ktheta"         K-theta model, k = [k1_psi, k2]
The last layer is the subgrade (half-space); its thickness is ignored.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Callable, Sequence
import numpy as np

PCF_TO_PCI = 1.0 / 12.0 ** 3
LBIN_TO_KNM = 1.0 / 224.81 * 39.37     # lb/in -> kN/m (as in MEAPA_MATFEA.m)


@dataclass
class Geogrid:
    l_in: float            # height above the bottom of the layer (NOT 0, NOT the thickness)
    delta_in: float        # total depth of influence (zone is +/- delta/2)
    M_kNm: float           # sheet stiffness, kN/m
    vg: float = 0.35       # geogrid Poisson's ratio
    phi_deg: float = 45.0  # friction angle of the unbound layer
    c_psi: float = 0.0     # cohesion of the unbound layer, psi
    eps0: float = 0.149    # NCHRP 1-50 plastic strain coefficients
    rho: float = 72.4
    beta: float = 0.247
    m: float = 1.72
    n: float = -2.16
    N: float = 1.0e6       # number of load cycles


@dataclass
class Layer:
    name: str
    type: str                                  # elastic | elastic_depth | universal | ktheta
    thickness_in: float = 0.0                  # ignored for the last layer
    E_psi: float = 0.0                         # elastic modulus or seed modulus (0 -> k1*Pa)
    nu: float = 0.35
    unit_weight_pcf: float = 130.0
    k: Sequence[float] = field(default_factory=list)
    K0: float = 0.5
    residual_psi: Optional[float] = None       # None -> 3 psi for nonlinear layers, 0 otherwise
    Mr_min_psi: Optional[float] = None         # None -> 1000 psi for nonlinear layers
    E_table: Optional[Sequence[Sequence[float]]] = None   # [[depth_in, E_psi], ...]
    Er_over_Ez: float = 0.99                   # anisotropy ratio used in the geogrid equation
    nu_hor: Optional[float] = None             # horizontal Poisson's ratio (geogrid eq.); None -> nu
    geogrids: List[Geogrid] = field(default_factory=list)

    # ---- derived -------------------------------------------------------------
    @property
    def is_elastic(self) -> bool:
        return self.type in ("elastic", "elastic_depth")

    @property
    def is_nonlinear(self) -> bool:
        return self.type in ("universal", "ktheta")

    @property
    def material_type(self) -> str:
        """MatFEA MaterialType string."""
        gg = "-GG" if self.geogrids else ""
        return {"elastic": "Elastic", "elastic_depth": "Elastic:MrFunc",
                "universal": "Nonlin:Universal" + gg, "ktheta": "Nonlin:KTheta" + gg}[self.type]

    def modulus_function(self) -> Optional[Callable[[np.ndarray], np.ndarray]]:
        """E(depth) for 'elastic_depth' layers (linear interpolation, linear extrapolation)."""
        if self.type != "elastic_depth":
            return None
        tab = np.asarray(self.E_table, float)
        z, E = tab[:, 0], tab[:, 1]
        order = np.argsort(z)
        z, E = z[order], E[order]

        def f(depth):
            depth = np.asarray(depth, float)
            out = np.interp(depth, z, E)
            if z.size > 1:                                   # linear extrapolation like interp1(...,'extrap')
                lo = depth < z[0]
                hi = depth > z[-1]
                out = np.where(lo, E[0] + (depth - z[0]) * (E[1] - E[0]) / (z[1] - z[0]), out)
                out = np.where(hi, E[-1] + (depth - z[-1]) * (E[-1] - E[-2]) / (z[-1] - z[-2]), out)
            return out
        return f


@dataclass
class Model:
    layers: List[Layer]
    pressure_psi: float = 120.0
    radius_in: float = 5.0
    name: str = ""
    width_in: Optional[float] = None           # None -> 40 * radius
    depth_in: Optional[float] = None           # None -> 50 * total thickness of the layers above the subgrade
    mesh_scale: float = 1.3
    mesh_import_file: Optional[str] = None     # MatFEA '.mesh' text file or a MATLAB dump JSON
    gauss_points: int = 16
    max_iter: int = 60
    reliability_pct: float = 95.0
    max_error_pct: float = 2.0
    atm_pressure_psi: float = 14.706
    scheme: str = "anderson"              # 'anderson' (robust) or 'picard' (MatFEA update)
    anderson_memory: int = 10
    anderson_beta: float = 0.8
    verbose: bool = True

    # ---- derived -------------------------------------------------------------
    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def thickness(self) -> np.ndarray:
        """Thicknesses of the layers above the subgrade (MatFEA geometry.LayerHeight)."""
        return np.array([L.thickness_in for L in self.layers[:-1]], float)

    @property
    def thickness_cum(self) -> np.ndarray:
        return np.cumsum(self.thickness)

    @property
    def width(self) -> float:
        return float(self.width_in) if self.width_in else 40.0 * self.radius_in

    @property
    def depth(self) -> float:
        return float(self.depth_in) if self.depth_in else 50.0 * float(self.thickness_cum[-1])

    @property
    def contact_load_lb(self) -> float:
        return self.pressure_psi * np.pi * self.radius_in ** 2

    def resolved(self):
        """Per-layer arrays in MatFEA form (seed modulus, floors, residual stress ...)."""
        Pa = self.atm_pressure_psi
        n = self.n_layers
        E0 = np.zeros(n); nu = np.zeros(n); nuh = np.zeros(n); gam = np.zeros(n)
        K0 = np.zeros(n); res = np.zeros(n); mrmin = np.full(n, np.nan); ratio = np.ones(n)
        kmat = np.full((4, n), np.nan)
        for i, L in enumerate(self.layers):
            nu[i] = L.nu
            nuh[i] = L.nu if L.nu_hor is None else L.nu_hor
            gam[i] = L.unit_weight_pcf * PCF_TO_PCI if L.unit_weight_pcf > 1 else L.unit_weight_pcf
            K0[i] = L.K0
            ratio[i] = L.Er_over_Ez
            E0[i] = L.E_psi
            if L.is_nonlinear:
                k = np.asarray(L.k, float)
                kmat[:k.size, i] = k
                mrmin[i] = L.Mr_min_psi if L.Mr_min_psi is not None else (
                    k[3] if k.size >= 4 and np.isfinite(k[3]) else 1000.0)
                res[i] = 3.0 if L.residual_psi is None else L.residual_psi
                if not E0[i] > 0:                                   # seed modulus (ff_run_MatFEA)
                    E0[i] = k[0] * Pa ** k[1] if L.type == "ktheta" else k[0] * Pa
                E0[i] = max(E0[i], mrmin[i])
            else:
                res[i] = 0.0 if L.residual_psi is None else L.residual_psi
        return dict(E0=E0, nu=nu, nu_hor=nuh, gamma_pci=gam, K0=K0, residual=res,
                    Mr_min=mrmin, Er_over_Ez=ratio, kmat=kmat,
                    mat_type=[L.material_type for L in self.layers])

    # ---- I/O --------------------------------------------------------------------
    def to_dict(self) -> dict:
        d = {"name": self.name,
             "load": {"pressure_psi": self.pressure_psi, "radius_in": self.radius_in},
             "layers": [],
             "domain": {"width_in": self.width_in, "depth_in": self.depth_in},
             "mesh": {"scale": self.mesh_scale, "import_file": self.mesh_import_file},
             "solver": {"gauss_points": self.gauss_points, "max_iter": self.max_iter,
                        "reliability_pct": self.reliability_pct, "max_error_pct": self.max_error_pct,
                        "atm_pressure_psi": self.atm_pressure_psi, "scheme": self.scheme,
                        "anderson_memory": self.anderson_memory, "anderson_beta": self.anderson_beta}}
        for L in self.layers:
            ld = asdict(L)
            ld["k"] = list(L.k)
            ld["geogrids"] = [asdict(g) for g in L.geogrids]
            d["layers"].append({k: v for k, v in ld.items() if v is not None and v != []})
        return d

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def model_from_dict(d: dict) -> Model:
    layers = []
    for ld in d["layers"]:
        ld = dict(ld)
        ggs = [Geogrid(**g) for g in ld.pop("geogrids", [])]
        ld.pop("is_elastic", None); ld.pop("is_nonlinear", None); ld.pop("material_type", None)
        layers.append(Layer(**ld, geogrids=ggs))
    load = d.get("load", {})
    dom = d.get("domain", {}) or {}
    mesh = d.get("mesh", {}) or {}
    sol = d.get("solver", {}) or {}
    return Model(layers=layers, name=d.get("name", ""),
                 pressure_psi=load.get("pressure_psi", 120.0), radius_in=load.get("radius_in", 5.0),
                 width_in=dom.get("width_in"), depth_in=dom.get("depth_in"),
                 mesh_scale=mesh.get("scale", 1.3), mesh_import_file=mesh.get("import_file"),
                 gauss_points=sol.get("gauss_points", 16), max_iter=sol.get("max_iter", 20),
                 reliability_pct=sol.get("reliability_pct", 95.0),
                 max_error_pct=sol.get("max_error_pct", 2.0),
                 atm_pressure_psi=sol.get("atm_pressure_psi", 14.706),
                 scheme=sol.get("scheme", "anderson"), anderson_memory=sol.get("anderson_memory", 10),
                 anderson_beta=sol.get("anderson_beta", 0.8))


def load_model(path: str) -> Model:
    with open(path) as f:
        return model_from_dict(json.load(f))
