"""Example inputs shown in the application's Examples menu (also written to examples/*.json)."""

_LAYERS_LEA = [
    {"name": "AC", "type": "elastic", "thickness_in": 5.0, "E_psi": 400000, "nu": 0.35, "unit_weight_pcf": 145},
    {"name": "Base", "type": "elastic", "thickness_in": 8.0, "E_psi": 40000, "nu": 0.40, "unit_weight_pcf": 135},
    {"name": "Subbase", "type": "elastic", "thickness_in": 12.0, "E_psi": 20000, "nu": 0.40, "unit_weight_pcf": 130},
    {"name": "Subgrade", "type": "elastic", "E_psi": 5000, "nu": 0.45, "unit_weight_pcf": 120},
]
_SOLVER = {"gauss_points": 16, "max_iter": 60, "reliability_pct": 95, "max_error_pct": 2.0, "atm_pressure_psi": 14.706}
_TAIL = {"domain": {"width_in": None, "depth_in": None}, "mesh": {"scale": 1.3, "import_file": None}, "solver": _SOLVER}


def _case(name, load, layers=None, **extra):
    d = {"name": name, "load": load, "layers": layers or _LAYERS_LEA}
    d.update(_TAIL)
    d.update(extra)
    return d


EXAMPLES = [
    {"id": "single", "label": "Single circle (linear elastic)",
     "data": _case("Single circle, 9,000 lb at 120 psi - AC 5 in / base 8 in / subbase 12 in",
                   {"type": "single", "pressure_psi": 120.0, "radius_in": 4.886})},
    {"id": "dual", "label": "Dual tires (two circles)",
     "data": _case("Dual tires, 2 x 4,500 lb at 120 psi, 13.5 in spacing",
                   {"type": "dual", "pressure_psi": 120.0, "radius_in": 3.455, "spacing_in": 13.5})},
    {"id": "wb445", "label": "Wide-base tire footprint (NGWB 445)",
     "data": _case("Wide-base 445 measured footprint, 9 ribs",
                   {"type": "footprint", "rib_width_in": 1.42,
                    "rib_lengths_in": [7.8, 7.8, 7.8, 7.8, 8.5, 8.2, 7.8, 7.3, 8.1],
                    "rib_pressures_psi": [115, 115, 115, 125, 125, 125, 115, 115, 115],
                    "groove_width_in": 0.0, "n_tires": 1})},
    {"id": "wb455", "label": "Wide-base tire footprint (NGWB 455)",
     "data": _case("Wide-base 455 measured footprint, 9 ribs",
                   {"type": "footprint", "rib_width_in": 1.34,
                    "rib_lengths_in": [6.2, 7.6, 7.9, 8.3, 8.4, 8.3, 8.4, 10.3, 6.3],
                    "rib_pressures_psi": [68, 103, 106, 113, 117, 113, 106, 103, 68],
                    "groove_width_in": 0.0, "n_tires": 1})},
    {"id": "dualfp", "label": "Dual-tire footprint (2 x 5 ribs)",
     "data": _case("Dual tires, measured footprint, 5 ribs per tire, 11.9 in spacing",
                   {"type": "footprint", "rib_width_in": 1.4,
                    "rib_lengths_in": [7.8, 8.7, 9.3, 8.8, 8.0],
                    "rib_pressures_psi": [100, 130, 125, 130, 100],
                    "groove_width_in": 0.0, "n_tires": 2, "tire_spacing_in": 11.9})},
    {"id": "sweep", "label": "Batch: AC thickness x AC modulus",
     "data": _case("Batch - single circle, AC thickness and modulus",
                   {"type": "single", "pressure_psi": 120.0, "radius_in": 4.886},
                   sweep={"AC.thickness_in": [3, 5, 7, 9], "AC.E_psi": [200000, 400000, 800000]})},
    {"id": "nonlinear", "label": "Single circle, stress-dependent base (FL-131487)",
     "data": _case("FL-131487 AC 3in / base 5in (universal) / subgrade 4000 psi - no geogrid",
                   {"type": "single", "pressure_psi": 120.0, "radius_in": 4.8},
                   layers=[
                       {"name": "AC", "type": "elastic", "thickness_in": 3.0, "E_psi": 400000, "nu": 0.2523, "unit_weight_pcf": 145},
                       {"name": "Base", "type": "universal", "thickness_in": 5.0, "k": [1650.0, 0.85, -0.15], "nu": 0.3,
                        "unit_weight_pcf": 135, "K0": 0.5, "residual_psi": 3.0, "Mr_min_psi": 1000},
                       {"name": "Subgrade", "type": "elastic", "E_psi": 4000, "nu": 0.35, "unit_weight_pcf": 120}])},
]
