"""MatLEA (layered elastic, Burmister) as the nartfem-apd response engine - the legacy path, unchanged:
one single-tire solution per climate case at the 18-kip reference tire, then
ff_superpose_strains (legacy f_njit_functions) for every axle type.  Load levels are exact
linear scalings for a linear structure, so ``levels`` costs nothing here."""
from __future__ import annotations
import time
import numpy as np

from .legacy_bridge import legacy_modules
from .engine_base import ResponseEngine, AXLE_KEYS

legacy_modules()
from MatLEA_MAIN_CODEMOD import MatLEA                     # noqa: E402
from f_njit_functions import ff_superpose_strains           # noqa: E402


class MatLEAEngine(ResponseEngine):
    name = "matlea"

    def __init__(self, ctx, options=None, log=None):
        super().__init__(ctx)
        self.stats = {"solves": 0, "time_s": 0.0}

    def single_tire(self, E_sub):
        c = self.ctx
        t0 = time.time()
        out = MatLEA(c["ZC_in_subM"], c["RC"], c["tirePressure"], c["CR_in"], np.asarray(E_sub, float),
                     c["PR_sub"], c["TH_sub"], None)
        self.stats["solves"] += 1; self.stats["time_s"] += time.time() - t0
        return out

    def responses(self, E_sub, levels=(1.0,)):
        c = self.ctx
        out = self.single_tire(E_sub)
        rz = np.vstack((out["r"], out["z"])).T
        res = {}
        for ax in AXLE_KEYS:
            exx, exy, exz, eyx, eyy, eyz, ezx, ezy, ezz, princ = ff_superpose_strains(
                c["RM_" + ax], c["theta_" + ax], c["ZC_in"], out["strain_r"], out["strain_t"],
                out["strain_z"], out["strain_rz"], rz)
            base = {"eps_xx": exx, "eps_yy": eyy, "eps_zz": ezz, "eps_princ": np.real(princ).ravel()}
            res[ax] = [{k: v * lev for k, v in base.items()} for lev in levels]
        self._last_single_tire = out
        return res
