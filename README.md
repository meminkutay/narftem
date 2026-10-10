# nartfem — nonlinear pavement finite element analysis in Python

`nartfem.axi` is a Python port of **MatFEA** (Abdollahi & Kutay, MATLAB): the axisymmetric
nonlinear finite element engine of MEAPA-MatFEA, including the stress-dependent unbound
layers (NCHRP 1-37A universal and K-theta models), the NCHRP 1-50 geogrid confinement model
and the September 2026 corrections. `nartfem.solid3d` is reserved for the 3D engine.

Requirements: Python ≥ 3.9, numpy, scipy ≥ 1.13 (modified-Akima interpolation), matplotlib (plots only).

```
nartfem/
  nartfem/axi/          the engine (one module per MatFEA step, see below)
  examples/             model JSON files (FL 3/5 in with and without geogrid, East Lansing 4/8 in)
  main.py        run one or more model files, print/plot results
  scripts/compare_with_matlab.py   compare with MatFEA on the identical mesh
  matlab/nartfem_json_to_Model.m   MATLAB: build the MatFEA Model struct from a nartfem JSON
  matlab/run_matfea_dump.m         MATLAB: run MatFEA on a nartfem JSON and dump mesh + results
```

## Run in Python

```bash
python main.py examples/FL_AC3_B5_SG4k_GG.json examples/FL_AC3_B5_SG4k_noGG.json --same-mesh --plot
```

or in code:

```python
from nartfem.axi import load_model, run
model = load_model("examples/FL_AC3_B5_SG4k_GG.json")
res   = run(model)                      # 1-10 s
res.layer_summary()                     # whole-layer / near-load / area-weighted moduli
res.interp("strain_t", 0.0, 2.99)       # any response at (r, depth) - for MEAPA
res.E_centerline(2, 5.5)                # modulus along the axis (MatFEA out.Emodels)
res.plot("Mr"); res.plot("delsig")      # element maps
```

For MEAPA-style superposition of several tires: `nartfem.axi.superpose.superpose_strains(res, RM, theta, ZC)`
(port of `ff_superpose_strains.m`).

## Run MatFEA (MATLAB) on the same input and compare

```matlab
addpath('<MatFEA V4 folder>'); addpath('<nartfem>/matlab');
run_matfea_dump('<nartfem>/examples/FL_AC3_B5_SG4k_GG.json');   % writes ..._matfea_dump.json
```
```bash
python scripts/compare_with_matlab.py examples/FL_AC3_B5_SG4k_GG.json
```

The Python run then uses the **MATLAB mesh** (imported from the dump), so the comparison is
mesh-identical: the first linear solve and the first modulus update must agree to round-off;
the converged solutions are compared for both iteration schemes (see below). A figure
`*_compare.png` is written next to the dump.

`nartfem_json_to_Model.m` applies exactly the conventions of `ff_run_MatFEA.m` (K0 = 0.5,
3 psi residual stress, 1000 psi floor or k4, seed modulus k1·Pa, pcf → pci, cohesion in psi,
sheet stiffness in kN/m), so a nartfem JSON is a complete, engine-independent description of
a MatFEA case.

## Model JSON

See the docstring of `nartfem/axi/model.py`. Layer types: `elastic`, `elastic_depth`
(E vs depth table = MatFEA `Elastic:MrFunc`), `universal` (k = [k1, k2, k3], k1 dimensionless),
`ktheta` (k = [k1_psi, k2]). Geogrids are listed inside their layer (`l_in` from the layer
bottom, `delta_in`, `M_kNm`, `vg`, `phi_deg`, `c_psi`, `eps0`, `rho`, `beta`, `m`, `n`, `N`).
The last layer is the subgrade (half-space). Units: in, psi, pcf, kN/m, degrees.

## Correspondence with MatFEA

| MatFEA (MATLAB)                      | nartfem.axi                         | notes |
|--------------------------------------|-------------------------------------|-------|
| ff_run_MatFEA.m (Model defaults)     | model.py `Model.resolved()`         | same conventions |
| DefineGenerateMesh.m                 | mesh.py `generate`                  | structured graded mesh with the MatFEA seed rules (not the recursive mesher); `load` imports a MatFEA mesh |
| DefineCalculations.m                 | elements.py `ElementData`           | vectorized B, D, K; 4×4 Gauss; π·r integration |
| DefineForceMatrix / Constrains / Solver | assembly.py                      | ring loads ÷ 2, same BCs, SuperLU |
| DefineResponses.m (NewMrCalculation) | responses.py + material.py          | overburden, K0, residual stress, tension clip, floor |
| DefineGeogrid.m                      | geogrid.py                          | ~60 radial points, bracketed root (brentq), makima, triangular decay, layer-clipped |
| DefineNonlinearity.m                 | responses.py `iterate`              | same error measure; see schemes below |
| MatFEA.m output block, LayerMrSummary | results.py                         | `E`, `E_centerline`, `layer_summary`, interpolants |
| ff_superpose_strains.m               | superpose.py                        | tensor shear, principal strains of the sum |

## Iteration scheme (important)

MatFEA updates the moduli by plain substitution (Picard): `Mr := Mr(stresses)`. For thin AC on a
soft subgrade (e.g. the FL 3 in / 5 in / 4 ksi case) this iteration **does not converge**: base
elements just outside the tire edge flip between the floor and ~10 ksi every iteration,
because with the no-tension clip the modulus map is extremely steep there (stiff → carries
horizontal tension → clipped → soft → sheds the load → stiff ...). The 2 % mean+1.645·std
criterion is then never met and MatFEA stops at MaxIteration with whatever moduli the last
iteration produced.

`nartfem.axi` therefore solves the modulus field with **Anderson mixing** on ln(Mr)
(`scheme = "anderson"`, default; memory 10, damping 0.8). It converges to a true fixed point
of the same equations (max element residual a few %, no oscillating elements) in 5–50
iterations. `scheme = "picard"` reproduces the MatFEA update for one-to-one comparisons.
When comparing converged results with MATLAB, differences therefore appear only where MatFEA
oscillates; the linear solve and the first update are identical.

## Verification (MATLAB R2026b, identical mesh, October 2026)

`scripts/compare_with_matlab.py` on the three examples (MatFEA mesh imported from the dump):

| case | linear solve | 1st modulus update | geogrid delta-sigma-3 | Picard path vs MatFEA |
|---|---|---|---|---|
| FL 3/5/4k, no geogrid | 6e-12 | 1e-6 | - | 7e-6 over all 60 iterations |
| FL 3/5/4k, geogrid at base bottom | 6e-12 | 9e-4 | 1 % | diverges late (MatFEA oscillates, final error 805 %) |
| EL 4/8/15k, geogrid at mid-base | 1e-12 | 2e-3 | 0.7 % | 1e-3 (both end at ~2.8 %) |

(max |difference| / max |reference|). The sub-percent differences with a geogrid come from the
scattered interpolation of the stresses onto the geogrid plane (MATLAB scatteredInterpolant vs
scipy LinearNDInterpolator); everything else is identical to solver round-off. MatFEA reports
converged = false for all three cases (errors 539 %, 805 %, 2.9 %); the Anderson scheme converges
in 28, 53 and 4 iterations.

Other checks: homogeneous half-space surface deflection 2.3 % below the Boussinesq value (finite
domain with fixed edges), axis stresses within 1-4 %. In the FL case the bottom of the base is in
horizontal tension (~27 psi) on the axis, so the tension clip removes the geogrid confinement
there and the AC-bottom strain is unchanged; the modulus changes only off-axis.

## Next: nartfem.solid3d

Hex-element 3D engine with several tires (no superposition), same material and geogrid
models (geogrid confinement evaluated on an (x, y) grid of the geogrid plane), quarter/half
symmetry, Anderson-mixed nonlinear iteration, pypardiso/scikit-sparse solver.
