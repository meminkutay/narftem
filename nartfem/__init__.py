"""nartfem - finite element analysis of flexible pavements in Python.

Sub-packages
------------
nartfem.axi      Axisymmetric nonlinear engine: a faithful port of MatFEA
                 (Abdollahi & Kutay, MATLAB) including the NCHRP 1-50 geogrid
                 confinement model and the September 2026 corrections.
nartfem.solid3d  (reserved) 3D solid-element engine with multiple tires.

Quick start
-----------
>>> from nartfem.axi import load_model, run
>>> model = load_model("examples/FL_AC3_B5_SG4k_GG.json")
>>> res   = run(model)
>>> res.layer_summary()
"""
from .axi import load_model, run, Model, Results   # noqa: F401

__version__ = "0.1.0"
