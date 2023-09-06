# This code is part of a Qiskit project.
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

import numpy as np

from pyscf.gto import Mole
from pyscf.lo import PipekMezey
from qiskit_nature.second_q.operators import ElectronicIntegrals

from . import OccupiedOrbitalPartitioning


class PySCFPipekMezeyPartitioning(OccupiedOrbitalPartitioning):
    """TODO."""

    def __init__(self, molecule: Mole) -> None:
        """TODO."""
        self.molecule = molecule

    def partition(
        self,
        overlap: ElectronicIntegrals,
        mo_coeff_occ: ElectronicIntegrals,
        num_bf: int,
        num_occ_fragment: tuple[int, int],
    ) -> tuple[ElectronicIntegrals, ElectronicIntegrals]:
        """TODO."""
        # alpha spin
        nocc_a = num_occ_fragment[0]
        nocc_b = (self.molecule.nelectron - sum(num_occ_fragment)) // 2

        pm = PipekMezey(self.molecule)
        mo = pm.kernel(mo_coeff_occ.alpha["+-"], verbose=4)

        nocc = mo.shape[1]
        pop = np.zeros((nocc, 2))
        for i in range(nocc):
            col = mo[:, i]
            dens = np.outer(col, col)
            PS = np.dot(dens, overlap.alpha["+-"])

            pop[i, 0] = np.trace(PS[:num_bf, :num_bf])
            pop[i, 1] = np.trace(PS[num_bf:, num_bf:])

        pop_order_1 = np.argsort(-1 * pop[:, 0])
        pop_order_2 = np.argsort(-1 * pop[:, 1])

        orbid_1 = pop_order_1[:nocc_a]
        orbid_2 = pop_order_2[:nocc_b]

        nao = self.molecule.nao
        fragment_1_alpha = np.zeros((nao, nocc_a))
        fragment_2_alpha = np.zeros((nao, nocc_b))

        for i in range(nocc_a):
            fragment_1_alpha[:, i] = mo[:, orbid_1[i]]
        for i in range(nocc_b):
            fragment_2_alpha[:, i] = mo[:, orbid_2[i]]

        fragment_1_beta = None
        fragment_2_beta = None
        if "+-" in mo_coeff_occ.beta:
            # beta spin
            nocc_a = num_occ_fragment[1]
            nocc_b = (self.molecule.nelectron - sum(num_occ_fragment)) // 2

            mo = pm.kernel(mo_coeff_occ.beta["+-"], verbose=4)

            nocc = mo.shape[1]
            pop = np.zeros((nocc, 2))
            for i in range(nocc):
                col = mo[:, i]
                dens = np.outer(col, col)
                # NOTE: even though this is the beta-spin case, the overlap only has alpha
                # components (because they are identical)
                PS = np.dot(dens, overlap.alpha["+-"])

                pop[i, 0] = np.trace(PS[:num_bf, :num_bf])
                pop[i, 1] = np.trace(PS[num_bf:, num_bf:])

            pop_order_1 = np.argsort(-1 * pop[:, 0])
            pop_order_2 = np.argsort(-1 * pop[:, 1])

            orbid_1 = pop_order_1[:nocc_a]
            orbid_2 = pop_order_2[:nocc_b]

            nao = self.molecule.nao
            fragment_1_beta = np.zeros((nao, nocc_a))
            fragment_2_beta = np.zeros((nao, nocc_b))

            for i in range(nocc_a):
                fragment_1_beta[:, i] = mo[:, orbid_1[i]]
            for i in range(nocc_b):
                fragment_2_beta[:, i] = mo[:, orbid_2[i]]

        return (
            ElectronicIntegrals.from_raw_integrals(
                fragment_1_alpha, h1_b=fragment_1_beta, validate=False
            ),
            ElectronicIntegrals.from_raw_integrals(
                fragment_2_alpha, h1_b=fragment_2_beta, validate=False
            ),
        )
