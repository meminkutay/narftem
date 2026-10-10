"""nartfem.axi - axisymmetric nonlinear pavement FE (port of MatFEA).

Pipeline (same order as MatFEA.m):
    model  -> mesh.generate / mesh.load       (DefineGenerateMesh / DefineLoadMesh)
           -> elements.precompute             (DefineCalculations)
           -> assembly.force, .constraints, .solve   (DefineForceMatrix / Constrains / Solver)
           -> responses.update                (DefineResponses, incl. geogrid.update = DefineGeogrid)
           -> nonlinearity.iterate            (DefineNonlinearity)
           -> results.Results                 (MatFEA.m output block)
"""
from .model import Model, Layer, Geogrid, load_model, model_from_dict
from .run import run
from .results import Results

__all__ = ["Model", "Layer", "Geogrid", "load_model", "model_from_dict", "run", "Results"]
