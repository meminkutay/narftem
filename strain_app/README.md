# nartfem Strain Calculator (`strain_app/`)

A small local application around `nartfem.axi` for people who should run nartfem without seeing
or installing the code: paste a model JSON, press Run, read the critical strains.

- `nartfem_strains.py` - the calculation (load definitions, superposition, critical strains, batch sweep)
- `app.py`, `app_page.py`, `app_examples.py` - the application: a local web page served on 127.0.0.1
- `examples/` - the inputs of the Examples menu
- `build_mac.sh`, `build_windows.bat` - make the one-file program for the student
- `HOW_TO_USE.txt` - the one-page instructions that go to the student
- `verification/` - comparison with MatLEA (`verify_vs_matlea.py`, `results.txt`)

## Try it from the source

    python strain_app/app.py            # opens the page in the browser
    python strain_app/app.py --selftest # runs every example once, prints the strains

## Build the program for the student

PyInstaller builds for the operating system it runs on, so build on the kind of computer the
student uses.

    macOS:    bash strain_app/build_mac.sh
    Windows:  double-click strain_app\build_windows.bat

Each script makes its own virtual environment, installs numpy / scipy / pyinstaller, builds
`dist/nartfem_strains(.exe)`, runs the self-test on the packaged file and copies
`HOW_TO_USE.txt` and `examples/` next to it. Send the student the `dist` folder (zipped).

What the student gets: one executable. There are no `.py` files in it; the modules are stored as
compiled bytecode inside the executable. That keeps the code away from anyone who is not
deliberately reverse-engineering it; bytecode can be decompiled by someone determined. If that
matters, compile `nartfem/` and `nartfem_strains.py` to native extension modules with Cython (or
build with Nuitka) before packaging.

The program is not code-signed: Windows shows "Windows protected your PC" (More info -> Run
anyway) and macOS asks for right-click -> Open the first time.

## Input

The nartfem.axi model JSON with an extended `load` block:

| type | entries |
|---|---|
| `single` (default) | `pressure_psi`, `radius_in` (or `load_lb`) |
| `dual` | `pressure_psi`, `radius_in` (or `load_lb` per tire), `spacing_in` |
| `footprint` | `rib_width_in`, `rib_lengths_in`, `rib_pressures_psi`, `groove_width_in` (0), `n_tires` (1 or 2), `tire_spacing_in` |
| `circles` | `circles`: list of `x_in`, `y_in`, `radius_in`, `pressure_psi` |

Optional: `"output": {"asphalt_bottom_layer": n}` and
`"sweep": {"AC.thickness_in": [...], "AC.E_psi": [...], "load.pressure_psi": [...]}` (full factorial, at most 400 runs).

## Method

- Footprint: every rib (length x width at one pressure) becomes a row of equal circles of the same
  area - the rule of `f_optimize_circles` in the FV_MatLEA_*_TireFootprint scripts, with the same
  rib positions (central rib at x = 0, neighbouring rows touching; `groove_width_in` adds a gap).
- One axisymmetric nartfem solution per distinct circle radius, on a domain wide enough for the
  whole footprint (the default 40 x radius would be 30 in for a rib circle), then superposition of
  the strain tensors on an x-y grid; the peaks are searched on that grid.
- Strain recovery: horizontal strains are sampled at the radial element centers. The vertical strain
  at the subgrade interface is extrapolated (quadratic) from the first three element rows, because
  the value inside a 4-node element is the strain at its mid-height. Without this the subgrade strain
  is 3-5 % low against layered elastic theory.
- Stress-dependent layers: single circle = the nonlinear nartfem.axi solution; identical circles
  (dual) = `nartfem.axi.multitire`; footprint = the nonlinear solution of the equivalent circle of
  each tire (same load and contact area) gives a modulus-depth profile under the tire, which is then
  frozen (`elastic_depth` layers) for the rib superposition. The last one is an approximation and is
  written in the log of every such run.

## Verification against MatLEA (linear structures)

`verification/results.txt`: nartfem is within about 2 % of converged layered elastic theory for the
tensile strains and within about 1-3 % for the subgrade strain, for single circles, dual circles and the
445 / 455 / dual-tire footprints.
