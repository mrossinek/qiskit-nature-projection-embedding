# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""TODO."""

from __future__ import annotations

import numpy as np

from pyscf.dft import KohnShamDFT
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.operators import ElectronicIntegrals
from qiskit_nature.second_q.properties import ElectronicDensity

from . import FockBuilder


class PySCFDFTBuilder(FockBuilder):
    """TODO."""

    def __init__(self, scf: KohnShamDFT) -> None:
        """TODO."""
        self.scf = scf

    def build(
        self,
        hamiltonian: ElectronicEnergy,
        density_a: ElectronicDensity,
        density_b: ElectronicDensity,
    ) -> tuple[ElectronicEnergy, float]:
        """TODO."""
        density_tot = density_a + density_b

        h_core = hamiltonian.electronic_integrals.one_body

        pyscf_rho_a = np.asarray(density_a.trace_spin()["+-"])
        pyscf_rho_tot = np.asarray(density_tot.trace_spin()["+-"])

        pyscf_fock_a = self.scf.get_fock(dm=pyscf_rho_a)
        pyscf_fock_tot = self.scf.get_fock(dm=pyscf_rho_tot)

        e_low_level_a = self.scf.energy_tot(dm=pyscf_rho_a)
        e_low_level_tot = self.scf.energy_tot(dm=pyscf_rho_tot)

        # TODO: support unrestricted spin cases
        fock_final = hamiltonian.fock(density_a)
        h_core_a = h_core.alpha["+-"]
        fock_delta = (pyscf_fock_tot - h_core_a) - (pyscf_fock_a - h_core_a)
        fock_final = ElectronicIntegrals.from_raw_integrals(
            fock_final.alpha["+-"] + fock_delta
        )

        e_tot = e_low_level_tot - e_low_level_a

        return fock_final, e_tot
