# This code is part of Qiskit.
#
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

from qiskit_nature.second_q.operators import ElectronicIntegrals


class OccupiedOrbitalPartitioning(ABC):
    """TODO."""

    @abstractmethod
    def partition(
        self,
        overlap: ElectronicIntegrals,
        mo_coeff_occ: ElectronicIntegrals,
        num_bf: int,
        num_occ_fragment: tuple[int, int],
    ) -> tuple[ElectronicIntegrals, ElectronicIntegrals]:
        """TODO."""
        raise NotImplementedError()