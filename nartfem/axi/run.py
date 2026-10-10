"""Orchestrator (MatFEA.m) and command-line entry point.

    python -m nartfem.axi.run examples/FL_AC3_B5_SG4k_GG.json [--out results.json] [--plot]
"""
from __future__ import annotations
import argparse
import time
import numpy as np
from . import mesh as meshmod
from .elements import ElementData, overburden
from .assembly import force_vector, fixed_dofs, Solver
from .responses import State, iterate
from .results import Results


def run(model, mesh=None, log=None) -> Results:
    """Run the axisymmetric nonlinear analysis for a Model; returns a Results object."""
    log = log or (print if model.verbose else (lambda *a, **k: None))
    t0 = time.time()
    if model.name:
        log("nartfem.axi: " + model.name)
    # 1. mesh
    if mesh is None:
        if model.mesh_import_file:
            mesh = meshmod.load(model, model.mesh_import_file)
        else:
            mesh = meshmod.generate(model)
    else:
        zif = np.unique(np.round(mesh.nodes[:, 1], 6))
        for zc in model.thickness_cum:
            if not np.any(np.abs(zif + zc) < 1e-6):
                raise ValueError("The given mesh has no node row at the interface depth %.3f in of this model; "
                                 "a shared mesh needs identical layer thicknesses." % zc)
        mesh = meshmod.relabel(model, mesh)
    log("  - mesh: %d nodes, %d elements (%s)" % (mesh.n_nodes, mesh.n_elements, mesh.source))
    # 2. element matrices
    res = model.resolved()
    ed = ElementData(mesh, model.gauss_points)
    ed.add_interface_points(np.concatenate([[0.0], model.thickness_cum, [mesh.depth]]))
    state = State(model, mesh, ed, res)
    state.gz = overburden(mesh, res, model.thickness)
    log("  - stiffness assembled (%.1f s)" % (time.time() - t0))
    # 3. load, BC, linear solve
    F = force_vector(mesh, model.pressure_psi, model.radius_in)
    solver = Solver(2 * mesh.n_nodes, fixed_dofs(mesh))
    state.u = solver.solve(state.K, F)
    # 4. responses (+ first modulus update) and nonlinear iterations
    state.responses()
    log("  - linear solution and first modulus update (%.1f s)" % (time.time() - t0))
    if np.any(state.nonlin):
        log("  - nonlinear iterations (%.0f%% reliability, %.1f%% tolerance):"
            % (model.reliability_pct, model.max_error_pct))
    iterate(state, solver, F, log=log)
    results = Results(model, mesh, state, time.time() - t0)
    log("  - done in %.1f s; layer mean moduli (psi): %s" % (results.elapsed_s,
        " ".join("%.0f" % e for e in results.E)))
    results.layer_summary(log=log)
    return results


def main(argv=None):
    from .model import load_model
    p = argparse.ArgumentParser(description="nartfem.axi - axisymmetric nonlinear pavement FE (MatFEA port)")
    p.add_argument("model_json")
    p.add_argument("--out", help="write results JSON")
    p.add_argument("--plot", action="store_true", help="show modulus and vertical strain maps")
    a = p.parse_args(argv)
    model = load_model(a.model_json)
    res = run(model)
    if a.out:
        res.save(a.out); print("results written to", a.out)
    if a.plot:
        import matplotlib.pyplot as plt
        res.plot("Mr"); res.plot("strain_z")
        if np.any(res.elem_delsig > 0):
            res.plot("delsig")
        plt.show()
    return res


if __name__ == "__main__":
    main()
