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

from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.operators import ElectronicIntegrals
from qiskit_nature.second_q.properties import ElectronicDensity

from . import FockBuilder


class HartreeFockBuilder(FockBuilder):
    """TODO."""

    def build(
        self,
        hamiltonian: ElectronicEnergy,
        density_a: ElectronicDensity,
        density_b: ElectronicDensity,
    ) -> tuple[ElectronicEnergy, float]:
        """TODO."""
        density_tot = density_a + density_b

        h_core = hamiltonian.electronic_integrals.one_body

        fock_a = hamiltonian.fock(density_a)
        fock_tot = hamiltonian.fock(density_tot)

        e_low_level = 0.5 * ElectronicIntegrals.einsum(
            {"ij,ij": ("+-", "+-", "")}, fock_tot + h_core, density_tot
        )
        e_low_level -= 0.5 * ElectronicIntegrals.einsum(
            {"ij,ij": ("+-", "+-", "")}, fock_a + h_core, density_a
        )

        fock_final = hamiltonian.fock(density_a)
        fock_final += (fock_tot - h_core) - (fock_a - h_core)

        return fock_final, (
            e_low_level.alpha.get("", 0.0)
            + e_low_level.beta.get("", 0.0)
            + e_low_level.beta_alpha.get("", 0.0)
        )
