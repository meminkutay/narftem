# nartfem.apd - asphalt pavement distress prediction

`nartfem.apd` predicts the distresses of an asphalt pavement over its design life (bottom-up and
top-down fatigue cracking, AC / base / subgrade rutting, thermal cracking, IRI, with reliability)
from a single JSON input.  The structural responses come from the nartfem nonlinear finite
element engine (`nartfem.axi`, iterated superposition of the axle-group tires, stress-dependent
unbound layers, geogrid confinement) or, for all-linear structures and cross-checks, from a
layered-elastic engine.  The climate, traffic, |E*| and transfer-function preprocessing is the
legacy UPDAPSAC code in `_UPDAPSAC_MinersLaw/`, which is imported unchanged.

Full documentation: `documentation/nartfem-apd_Documentation.docx`.

| module | role |
|---|---|
| `schema.py` | the `nartfem-apd-1` input format (docstring = specification), validation, engine choice |
| `convert_legacy.py` | legacy UPDAPSAC JSON -> `nartfem-apd-1` |
| `legacy_bridge.py` | `nartfem-apd-1` -> legacy `inps` for the unchanged preprocessor; MATLAB `.hcm` climate files -> cached `.csv.gz` |
| `engine_base.py` | engine interface, tire positions per axle group, analysis points |
| `engine_matlea.py` | layered-elastic engine (reference; all-linear structures) |
| `engine_nartfem.py` | nartfem FE engine: iterated superposition, mesh/stiffness reuse, exact in-element strains |
| `distress.py` | Miner's-law loop over months x temperature quintiles x axle groups x load bins; monotone plastic strain |
| `run.py` | driver (CONFIG block / CLI) |

Scripts (`scripts/`): `apd_batch.py` (all inputs of a folder, in parallel), `prep_apd_inputs_v1.py`
(thickness / field sweeps from a template), `apd_plot_output.py`, `apd_plot_compare.py`,
`apd_compare_engines.py`, `apd_make_examples.py`.

```bash
python nartfem/apd/run.py                              # CONFIG block (PyCharm)
python -m nartfem.apd.run examples/apd/FL_AC3_B5_SG4k_GG.apd.json --engine nartfem --years 20 --workers 4
python -m nartfem.apd.convert_legacy legacy_input.json  # legacy -> nartfem-apd-1
python scripts/apd_batch.py                             # folder of inputs
```

Key options (`analysis.response_engine` in the input, overridable from CONFIG/CLI): `engine`
(auto | nartfem | matlea), `load_scaling` (linear | power), `shear` (legacy | tensor), `n_phi`,
`workers`, `mesh_scale`, `subgrade_split_in`, `warm_start`, `rut_monotone`.

Requirements: numpy, scipy, numba, pandas (legacy preprocessor), matplotlib, nartfem.
