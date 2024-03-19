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

from abc import ABC, abstractmethod

from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.properties import ElectronicDensity


class FockBuilder(ABC):
    """TODO."""

    @abstractmethod
    def build(
        self,
        hamiltonian: ElectronicEnergy,
        density_a: ElectronicDensity,
        density_b: ElectronicDensity,
    ) -> tuple[ElectronicEnergy, float]:
        """TODO."""
        raise NotImplementedError()
