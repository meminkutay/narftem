# meapa - MEAPA (Python) with a clean input format and nartfem as response engine

`meapa/` sits next to the legacy `_UPDAPSAC_MinersLaw/` package and **does not modify it**:
the legacy preprocessor (traffic, sublayering, climate, |E\*| at the temperature quintiles,
thermal cracking), the legacy transfer functions (`f_njit_functions.py`) and the legacy
reliability post-processor are imported and called unchanged.  What is new:

| module | role |
|---|---|
| `schema.py` | the meapa-2 input format (docstring = specification), validation, engine choice |
| `convert_legacy.py` | legacy `exampleinputs/*.json` -> meapa-2 (CONFIG block, or CLI) |
| `legacy_bridge.py` | meapa-2 -> legacy `inps` dict for `UPDAPSAC_STEP1`; MATLAB `.hcm` climate files -> cached `.csv.gz` |
| `engine_base.py` | engine interface, tire positions per axle type, Step-1 context |
| `engine_matlea.py` | MatLEA + legacy `ff_superpose_strains` (reference, all-linear structures) |
| `engine_nartfem.py` | `nartfem.axi` iterated superposition with mesh/stiffness reuse, exact in-element strains at MEAPA's points |
| `distress.py` | the Miner's-law loop of `f_struc_resp_ACGB` with an engine callback and explicit strain-vs-load |
| `run.py` | end-to-end driver (CONFIG block / CLI) |

Scripts: `scripts/meapa_make_examples.py` (meapa-2 inputs for the nartfem example structures),
`scripts/meapa_compare_engines.py` (tables + figure from the run outputs).

```bash
python -m meapa.convert_legacy                                       # legacy -> examples/meapa/*.meapa2.json
python -m meapa.run examples/meapa/FL_AC3_B5_SG4k_GG.meapa2.json     # engine from the input ("auto")
python -m meapa.run examples/meapa/FL_AC3_B5_SG4k_GG.meapa2.json --engine matlea --years 5
python -m meapa.run legacy_input.json                                 # legacy files are converted on the fly
python scripts/meapa_compare_engines.py
```
All scripts run from PyCharm with their CONFIG block (arguments optional).  Requirements: those
of the legacy package (numpy, scipy, numba, pandas, boto3) + nartfem.

## 1. Map of the legacy code (what `meapa` reuses and where it plugs in)

```
UPDAPSACPY.UPCAPSACRunOneFile
 ├─ UPDAPSAC_STEP1(inps)                      <- reused unchanged (via legacy_bridge.to_legacy_inps)
 │    processTrafficData  -> NAij_{Si,Ta,Tr,Qd}twMoQnt[month, load bin]  axle passes per quintile
 │    ff_sublayer_MEPDGmethod -> TH_sub (AC 0.5/0.5/1/1.. in, others 2 in), LayNo_sub
 │    analysis depths ZC: 0, sublayer mid-depths, 0.99 h_AC (BU), 0.5 in (TD), 1.001 h, h+6 (subgrade)
 │    f_calc_freqMatrix -> loading frequency f(z, load) and Ncorr (axle-group pass correction)
 │    f_analysis_locations -> X (10 across), Y (per axle group), tires at (+-Sx/2, k*Sy); RM, theta
 │    CR_in = sqrt(18000/4/(pi p))  reference tire (4500 lb) - all loads use this radius
 │    FiniteDiffDempsey (MCLIM) -> Tsubdum_M[month, sublayer, 5 quintiles]
 │    f_calculate_Estar_atQuintiles -> Esubdum_M[month, sublayer, quintile]  (|E*|(T, f) + GAS aging;
 │                                     unbound sublayers = constant layerMR)
 │    f_calc_TCcrack -> TC_mo; f_process_calib_coeffs -> C_bu, CH_bu, kzrut, ...; wander X_Wander
 ├─ UPDAPSAC_STEP2.f_struc_resp_ACGB         <- replaced by meapa.distress.run_miners (same math)
 │    per (month, quintile): MatLEA single tire -> ff_superpose_strains per axle type -> critical
 │    strains (BU: eps_xx/eps_yy at 0.99h, TD: max principal at z<=0.5, rut: -eps_zz per sublayer)
 │    -> eps(P) = eps_ref * P/P_ref -> Nf, Miner's damage with wander, MEPDG AC rut (strain
 │    hardening), Tseng-Lytton base/subgrade rut, sigmoid transfer functions, IRI
 └─ f_calc_distress_atrel                     <- reused unchanged
```

## 2. Response engines

`analysis.response_engine.engine`: `auto` (MatLEA when all layers are `elastic`/`elastic_depth`
without geogrids, otherwise nartfem), `matlea`, `nartfem`.

### MatLEA (`engine_matlea.py`)
Exactly the legacy path.  With `engine = matlea` and `load_scaling = linear` the new loop
reproduces `f_struc_resp_ACGB` to round-off (max relative difference 4e-14 on all monthly
distresses, FL case, 1 year).

### nartfem (`engine_nartfem.py`)
* Model: AC sublayers of MEAPA become separate elastic layers (each with its own |E\*|(T, f) of
  the climate case); stress-dependent unbound layers and geogrids as in the structure block
  (nartfem layer schema as-is); subgrade = last layer; tire = MEAPA's reference tire (p, CR_in).
* Per climate case and axle type: `run_multitire`-type iterated superposition (azimuth-averaged
  Mr from the superposed stresses of all tires of the axle group, Anderson mixing), i.e. 4
  nonlinear solves per case.  Mesh, B matrices, solver pattern and force vector are built once;
  AC element stiffnesses are rescaled when |E\*| changes; the unbound modulus field of the
  previous case is the warm start (2-3 iterations per case instead of 3-6).
* Responses: exact in-element strains (`StructuredEvaluator`, vectorized version of
  `Results.fe_at`) at MEAPA's (x, y, z) points, rotated and superposed per tire, maximum
  principal strain from the 3-D tensor.  Points on an interior interface are taken in the
  element below (what `MatLEA_Sub01_Find_LayNo` does).
* `subgrade_split_in = 6` (default): the subgrade gets a node row at h + 6 in (a 6-in top layer
  with identical properties).  MEAPA's subgrade rut model uses the ratio of the strains at h and
  h + 6 in (k = ln(eps_0/eps_6)/6, rut ~ eps_0/k) and is very sensitive to eps_6; inside a coarse
  subgrade element eps_6 moved by +-8 % with the mesh scale (and with the mesh changes caused by
  a geogrid plane), which showed up as a spurious +30 % "geogrid effect" on subgrade rut.  On the
  node row the value is mesh-independent (1.242 vs 1.244e-3 for mesh scale 1.3 vs 0.7, FL case).
  AC-bottom and subgrade-top strains are mesh-converged to < 1 % at the default scale.
* `shear = legacy` (default): the engineering shear gamma_rz is put in the strain tensor as
  MEAPA's `ff_superpose_strains` does (it affects only the maximum principal strain, i.e.
  top-down cracking; the tensor shear gamma/2 gives ~45 % lower principal strains at depth where
  shear dominates).  `shear = tensor` is the mechanically correct option.
* `load_scaling = linear` (default, MEAPA): one solution per case, strains proportional to the
  axle load.  `load_scaling = power`: solutions at `power_levels` x the reference pressure and
  log-log interpolation of the critical strains over the axle load.
* `workers > 1`: the climate cases are solved in a process pool (chunks of consecutive cases,
  so the warm start still applies), then the sequential Miner's loop runs on the cached responses.

Timing (this container, 2 cores, 20-year design life = 1200 cases): MatLEA loop ~10 min
(MatLEA itself ~100 s; the rest is the legacy njit code running without numba here);
nartfem ~14 min without geogrid, ~35 min with a geogrid (the geogrid confinement is re-evaluated
in every iteration), single process.

### Unchanged by design (documented, not "fixed")
* MEAPA uses the layer's resilient modulus (`layerMR` / `E_psi`) in the Tseng-Lytton water
  content term (`ErutBASE`, `ErutSUBG`) and in Nf (`EbuAC`); the nartfem engine keeps these
  inputs, only the strains come from the nonlinear solution.
* The legacy Python code ignores `UNBOUNDNONLIN` / `nonlinGGJSON`; with the legacy reader such a
  layer gets a dummy modulus of 0 in the frequency / Ncorr calculation.  The converter maps the
  layer to role `BASE` (dummy modulus of a base), which changes `Ncorr` at base/subgrade depth
  slightly (subgrade rut -3 % in the FL example) - this is the only difference between running
  a legacy file through `UPDAPSACPY` and through `meapa.run --engine matlea`.
* `f_construct_EstarMC` reuses one `layer` dict for all AC layers (fine for one AC layer).

## 3. Input format

See the docstring of `schema.py`.  The `structure.layers` entries are nartfem layers (`type`,
`thickness_in`, `E_psi`, `nu`, `unit_weight_pcf`, `k`, `K0`, `residual_psi`, `Mr_min_psi`,
`E_table`, `geogrids` ...) plus `role`, `thermal` and, for AC layers, `ac` (|E\*| / |G\*| data or
master-curve coefficients, Va, Vbeff, Tref, RTFO, IDT).  `convert_legacy.py` documents the
mapping of the legacy keys (and the assumption that `unbGG.a` is the geogrid on/off flag).

## 4. Verification

See `examples/meapa/results/comparison.md` (tables) and `comparison.png`, produced by
`scripts/meapa_compare_engines.py` from the 20-year runs of the four cases in `examples/meapa/`.
