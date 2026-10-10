"""3D model definition.  Layers and geogrids are the same dataclasses as nartfem.axi (so an
axisymmetric JSON's "layers" block can be reused verbatim); the load block lists the tires.

JSON (new / changed blocks relative to the axisymmetric file)::

    "tires": [ {"x_in": 6.75, "y_in": 0.0, "load_lb": 4500, "pressure_psi": 120, "shape": "circle"},
               {"x_in": -6.75, "y_in": 0.0, "load_lb": 4500, "pressure_psi": 120} ],
    "symmetry": "quarter",            # "quarter" (x>=0, y>=0), "half_x" (x>=0), "half_y" (y>=0), "none"
    "domain3d": {"x_max_in": null, "y_max_in": null, "depth_in": null},   # defaults: 30a, 30a, 50H
    "mesh3d":   {"fine_in": 0.5, "coarse_ratio": 1.0, "z_scale": 1.3},   # in-plane size under the tires, z grading
    "solver":   {... as axi ..., "backend": "auto"}                       # auto | pardiso | pyamg | splu | cg

Tires are given in full-model coordinates; with symmetry only the part inside the modeled
quadrant is loaded (a tire centered on a symmetry plane is cut in half, correctly).  The axle
travels in the y direction; x is transverse (dual spacing is in x).
"""
from __future__ import annotations
import json
import math
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import numpy as np
from ..axi.model import Layer, Geogrid, Model, model_from_dict as _axi_from_dict


@dataclass
class Tire:
    x_in: float
    y_in: float
    load_lb: float = 4500.0
    pressure_psi: float = 120.0
    shape: str = "circle"            # "circle" or "square" (equal area)

    @property
    def radius_in(self) -> float:
        return math.sqrt(self.load_lb / (self.pressure_psi * math.pi))

    @property
    def half_side_in(self) -> float:
        return math.sqrt(self.load_lb / self.pressure_psi) / 2


@dataclass
class Model3D:
    layers: List[Layer]
    tires: List[Tire]
    name: str = ""
    symmetry: str = "quarter"
    x_max_in: Optional[float] = None
    y_max_in: Optional[float] = None
    depth_in: Optional[float] = None
    fine_in: float = 0.5             # in-plane element size under/around the tires
    coarse_ratio: float = 1.0        # multiplies the outer coarse sizes
    z_scale: float = 1.3             # vertical grading scale (nartfem.axi mesh_scale rule)
    gauss_points: int = 8
    max_iter: int = 60
    reliability_pct: float = 95.0
    max_error_pct: float = 5.0           # MatFEA criterion (mean + 1.645 std of |dMr/Mr| in the active zone)
    max_error_vol_pct: float = 1.5       # volume-weighted mean |dMr/Mr| in the active zone (either one stops)
    atm_pressure_psi: float = 14.706
    scheme: str = "anderson"
    anderson_memory: int = 10
    anderson_beta: float = 0.8
    backend: str = "auto"
    verbose: bool = True

    # ---- derived (same conventions as axi.Model) --------------------------------
    @property
    def n_layers(self): return len(self.layers)

    @property
    def thickness(self): return np.array([L.thickness_in for L in self.layers[:-1]], float)

    @property
    def thickness_cum(self): return np.cumsum(self.thickness)

    @property
    def radius_ref(self) -> float:
        """Reference contact radius (largest tire) for default domain sizes."""
        return max(t.radius_in for t in self.tires)

    @property
    def x_extent(self) -> float:
        return float(self.x_max_in) if self.x_max_in else max(abs(t.x_in) for t in self.tires) + 30 * self.radius_ref

    @property
    def y_extent(self) -> float:
        return float(self.y_max_in) if self.y_max_in else max(abs(t.y_in) for t in self.tires) + 30 * self.radius_ref

    @property
    def depth(self) -> float:
        return float(self.depth_in) if self.depth_in else 50.0 * float(self.thickness_cum[-1])

    @property
    def x_min(self): return 0.0 if self.symmetry in ("quarter", "half_x") else -self.x_extent

    @property
    def y_min(self): return 0.0 if self.symmetry in ("quarter", "half_y") else -self.y_extent

    def axi_model(self) -> Model:
        """Equivalent axisymmetric Model (first tire) - used by Model.resolved() and comparisons."""
        t = self.tires[0]
        return Model(layers=self.layers, pressure_psi=t.pressure_psi, radius_in=t.radius_in, name=self.name,
                     mesh_scale=self.z_scale, gauss_points=16, max_iter=self.max_iter,
                     reliability_pct=self.reliability_pct, max_error_pct=self.max_error_pct,
                     atm_pressure_psi=self.atm_pressure_psi, scheme=self.scheme,
                     anderson_memory=self.anderson_memory, anderson_beta=self.anderson_beta, verbose=False)

    def resolved(self):
        return self.axi_model().resolved()

    def to_dict(self):
        d = self.axi_model().to_dict()
        d.pop("load"); d.pop("domain"); d.pop("mesh")
        d["tires"] = [asdict(t) for t in self.tires]
        d["symmetry"] = self.symmetry
        d["domain3d"] = {"x_max_in": self.x_max_in, "y_max_in": self.y_max_in, "depth_in": self.depth_in}
        d["mesh3d"] = {"fine_in": self.fine_in, "coarse_ratio": self.coarse_ratio, "z_scale": self.z_scale}
        d["solver"]["backend"] = self.backend; d["solver"]["gauss_points"] = self.gauss_points
        d["solver"]["max_error_pct"] = self.max_error_pct; d["solver"]["max_error_vol_pct"] = self.max_error_vol_pct
        return d

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def model3d_from_dict(d: dict) -> Model3D:
    base = _axi_from_dict({**d, "load": d.get("load", {"pressure_psi": 120, "radius_in": 5})})
    tires = [Tire(**t) for t in d["tires"]]
    dom = d.get("domain3d", {}) or {}
    m3 = d.get("mesh3d", {}) or {}
    sol = d.get("solver", {}) or {}
    return Model3D(layers=base.layers, tires=tires, name=d.get("name", ""),
                   symmetry=d.get("symmetry", "quarter"),
                   x_max_in=dom.get("x_max_in"), y_max_in=dom.get("y_max_in"), depth_in=dom.get("depth_in"),
                   fine_in=m3.get("fine_in", 0.5), coarse_ratio=m3.get("coarse_ratio", 1.0), z_scale=m3.get("z_scale", 1.3),
                   gauss_points=sol.get("gauss_points", 8), max_iter=sol.get("max_iter", 60),
                   reliability_pct=sol.get("reliability_pct", 95.0), max_error_pct=sol.get("max_error_pct", 5.0),
                   max_error_vol_pct=sol.get("max_error_vol_pct", 1.5),
                   atm_pressure_psi=sol.get("atm_pressure_psi", 14.706), scheme=sol.get("scheme", "anderson"),
                   anderson_memory=sol.get("anderson_memory", 10), anderson_beta=sol.get("anderson_beta", 0.8),
                   backend=sol.get("backend", "auto"))


def load_model3d(path: str) -> Model3D:
    with open(path) as f:
        return model3d_from_dict(json.load(f))
