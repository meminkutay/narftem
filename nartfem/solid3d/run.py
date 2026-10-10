"""Orchestrator for the 3D model."""
from __future__ import annotations
import time
import numpy as np
from . import mesh as meshmod
from .elements import ElementData3D
from .material import overburden
from .assembly import tire_loads, fixed_dofs, Solver3D
from .responses import State3D, iterate
from .results import Results3D


def run3d(model, mesh=None, log=None) -> Results3D:
    log = log or (print if model.verbose else (lambda *a, **k: None))
    t0 = time.time()
    if model.name:
        log("nartfem.solid3d: " + model.name)
    if mesh is None:
        mesh = meshmod.generate(model)
    nx, ny, nz = mesh.shape
    log("  - mesh: %d x %d x %d = %d hex elements, %d nodes, %d dofs (%s symmetry)"
        % (nx, ny, nz, mesh.n_elements, mesh.n_nodes, 3 * mesh.n_nodes, model.symmetry))
    res = model.resolved()
    ed = ElementData3D(mesh, model.gauss_points)
    state = State3D(model, mesh, ed, res, log=log)
    state.gz = overburden(mesh, res, model.thickness)
    F = tire_loads(mesh, model)
    log("  - tire loads: %.0f lb in the model (%d tires)" % (-F.sum(), len(model.tires)))
    solver = Solver3D(3 * mesh.n_nodes, fixed_dofs(mesh, model), model.backend, log=log)
    state.u = solver.solve(state.K, F)
    log("  - linear solve %.1f s" % solver.last_time)
    state.responses()
    if state.nl.size:
        log("  - nonlinear iterations (%s, %.0f%% reliability, %.1f%% tolerance):" % (model.scheme, model.reliability_pct, model.max_error_pct))
    iterate(state, solver, F, log=log)
    r = Results3D(model, mesh, state, time.time() - t0)
    log("  - done in %.0f s" % r.elapsed_s)
    r.layer_summary(log=log)
    return r
