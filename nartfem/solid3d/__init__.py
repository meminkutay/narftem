"""nartfem.solid3d - 3D solid-element nonlinear pavement FE with several tires.

Same layer, material (universal / K-theta, residual stress, floor) and NCHRP 1-50 geogrid
models as nartfem.axi, but every tire of an axle group is in the model: no superposition.

    from nartfem.solid3d import load_model3d, run3d
    model = load_model3d("examples/FL_AC3_B5_SG4k_GG_dual3d.json")
    res   = run3d(model)
    res.fe_at("strain_x", x, y, depth)          # exact in-element values
    res.plot_cut("Mr", plane="xz", y=0.0)       # vertical cut through the tires

Pipeline: mesh3d.generate -> elements3d (hex-8, 2x2x2 Gauss) -> loads (tire patches) ->
assembly (symmetry/roller BCs, solver backends) -> responses (Mr update, geogrid plane) ->
Anderson-mixed nonlinear iteration -> Results3D.
"""
from .model import Model3D, Tire, load_model3d, model3d_from_dict
from .run import run3d
from .results import Results3D

__all__ = ["Model3D", "Tire", "load_model3d", "model3d_from_dict", "run3d", "Results3D"]
