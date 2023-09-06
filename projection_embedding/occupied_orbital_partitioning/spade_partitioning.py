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

import logging
from functools import partial

import numpy as np

from qiskit_nature.second_q.operators import ElectronicIntegrals

from . import OccupiedOrbitalPartitioning
from ..utils import split_elec_ints_per_spin

logger = logging.getLogger(__name__)


class SPADEPartitioning(OccupiedOrbitalPartitioning):
    """TODO."""

    def partition(
        self,
        overlap: ElectronicIntegrals,
        mo_coeff_occ: ElectronicIntegrals,
        num_bf: int,
        num_occ_fragment: tuple[int, int],
    ) -> tuple[ElectronicIntegrals, ElectronicIntegrals]:
        """TODO."""
        logger.info("")
        logger.info("Doing SPADE partitioning")
        logger.info("D. CLaudino and N. Mayhall JCTC 15, 1053 (2019)")
        logger.info("")

        # 1. use symmetric orthogonalization on the overlap matrix
        symm_orth = SPADEPartitioning.symmetric_orthogonalization(overlap)

        # 2. change the MO basis to be orthogonal and reasonably localized
        mo_coeff_tmp = ElectronicIntegrals.einsum(
            {"ij,jk->ik": ("+-",) * 3}, symm_orth, mo_coeff_occ, validate=False
        )

        # 3. select the active sector
        mo_coeff_tmp, _ = mo_coeff_tmp.split(np.vsplit, [num_bf], validate=False)

        # 4. use SVD to find the final rotation matrix
        _, _, rot = ElectronicIntegrals.apply(
            partial(np.linalg.svd, full_matrices=True),
            mo_coeff_tmp,
            multi=True,
            validate=False,
        )
        rot_t = ElectronicIntegrals.apply(np.transpose, rot, validate=False)

        nocc_a_alpha, nocc_a_beta = num_occ_fragment
        left, right = split_elec_ints_per_spin(
            rot_t, np.hsplit, [nocc_a_alpha], [nocc_a_beta]
        )

        return (
            ElectronicIntegrals.einsum(
                {"ij,jk->ik": ("+-",) * 3}, mo_coeff_occ, left, validate=False
            ),
            ElectronicIntegrals.einsum(
                {"ij,jk->ik": ("+-",) * 3}, mo_coeff_occ, right, validate=False
            ),
        )

    @staticmethod
    def symmetric_orthogonalization(matrix: ElectronicIntegrals) -> ElectronicIntegrals:
        """TODO."""
        eigval, eigvec = ElectronicIntegrals.apply(
            np.linalg.eigh, matrix, multi=True, validate=False
        )
        eigval = ElectronicIntegrals.apply(
            lambda arr: np.diag(np.sqrt(arr)), eigval, validate=False
        )
        return ElectronicIntegrals.einsum(
            {"ik,kj,lj->il": ("+-",) * 4}, eigvec, eigval, eigvec
        )
