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

import logging
from functools import partial

import numpy as np

from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.operators import ElectronicIntegrals, PolynomialTensor

from . import VirtualOrbitalLocalization

logger = logging.getLogger(__name__)


class ConcentricLocalization(VirtualOrbitalLocalization):
    """TODO."""

    def __init__(self, zeta: int = 1) -> None:
        """TODO."""
        self.zeta = zeta

    def localize(
        self,
        overlap_pb_wb: ElectronicIntegrals,
        projection_basis: ElectronicIntegrals,
        mo_coeff_vir: ElectronicIntegrals,
        num_bf: int,
        fock: ElectronicEnergy,
    ) -> tuple[ElectronicIntegrals, tuple[int, int], tuple[int, int]]:
        """TODO."""
        logger.info("")
        logger.info("Doing concentric location and truncation of virtual space")
        logger.info("    Concentric localization and truncation for virtuals    ")
        logger.info("     D. Claudino and N. Mayhall, JCTC, 15, 6085 (2019)     ")
        logger.info("")

        # S^{-1} in paper
        overlap_a_pb_inv = ElectronicIntegrals.apply(
            np.linalg.inv, projection_basis, validate=False
        )

        # C'_{vir} in paper
        mo_coeff_vir_pb = ElectronicIntegrals.einsum(
            {"ij,jk,kl->il": ("+-",) * 4},
            overlap_a_pb_inv,
            overlap_pb_wb,
            mo_coeff_vir,
            validate=False,
        )

        # Eq. (10a)
        einsummed = ElectronicIntegrals.einsum(
            {"ji,jk,kl->il": ("+-",) * 4},
            mo_coeff_vir_pb,
            overlap_pb_wb,
            mo_coeff_vir,
            validate=False,
        )
        _, _, v_t = ElectronicIntegrals.apply(
            partial(np.linalg.svd, full_matrices=True),
            einsummed,
            multi=True,
            validate=False,
        )

        # Eq. (10b)
        v_t_t = ElectronicIntegrals.apply(np.transpose, v_t, validate=False)
        v_span, v_kern = v_t_t.split(np.hsplit, [num_bf], validate=False)

        # Eq. (10c)
        mo_coeff_vir_new = ElectronicIntegrals.einsum(
            {"ij,jk->ik": ("+-",) * 3}, mo_coeff_vir, v_span, validate=False
        )

        # Eq. (10d)
        mo_coeff_vir_kern = ElectronicIntegrals.einsum(
            {"ij,jk->ik": ("+-",) * 3}, mo_coeff_vir, v_kern, validate=False
        )

        mo_coeff_vir_cur = mo_coeff_vir_new

        for _ in range(self.zeta - 1):
            # mo_coeff_vir_new is the working variable
            fock_cur_kern = ElectronicIntegrals.einsum(
                {"ji,jk,kl->il": ("+-",) * 4},
                mo_coeff_vir_cur,
                fock,
                mo_coeff_vir_kern,
                validate=False,
            )

            _, _, r_t = ElectronicIntegrals.apply(
                partial(np.linalg.svd, full_matrices=True),
                fock_cur_kern,
                multi=True,
                validate=False,
            )
            r_t_t = ElectronicIntegrals.apply(np.transpose, r_t, validate=False)

            # update
            mo_coeff_vir_kern_ncols = mo_coeff_vir_kern.alpha["+-"].shape[1]
            mo_coeff_vir_cur_ncols = mo_coeff_vir_cur.alpha["+-"].shape[1]

            if mo_coeff_vir_kern_ncols > mo_coeff_vir_cur_ncols:
                r_t_t_left, r_t_t_right = r_t_t.apply(
                    np.hsplit, [mo_coeff_vir_cur_ncols], validate=False
                )
                mo_coeff_vir_cur = ElectronicIntegrals.einsum(
                    {"ij,jk->ik": ("+-",) * 3},
                    mo_coeff_vir_kern,
                    r_t_t_left,
                    validate=False,
                )
                mo_coeff_vir_kern = ElectronicIntegrals.einsum(
                    {"ij,jk->ik": ("+-",) * 3},
                    mo_coeff_vir_kern,
                    r_t_t_right,
                    validate=False,
                )
            else:
                mo_coeff_vir_cur = ElectronicIntegrals.einsum(
                    {"ij,jk->ik": ("+-",) * 3}, mo_coeff_vir_kern, r_t_t, validate=False
                )
                mo_coeff_vir_kern = ElectronicIntegrals.from_raw_integrals(
                    np.zeros_like(mo_coeff_vir_kern.alpha["+-"]),
                    h1_b=None
                    if mo_coeff_vir_kern.beta.is_empty()
                    else np.zeros_like(mo_coeff_vir_kern.beta["+-"]),
                    validate=False,
                )

            mo_coeff_vir_new = ElectronicIntegrals.stack(
                np.hstack, (mo_coeff_vir_new, mo_coeff_vir_cur), validate=False
            )

        # post-processing step
        logger.info(
            "Pseudocanonicalizing the selected and excluded virtuals separately"
        )

        einsummed = ElectronicIntegrals.einsum(
            {"ji,jk,kl->il": ("+-",) * 4},
            mo_coeff_vir_new,
            fock,
            mo_coeff_vir_new,
            validate=False,
        )

        _, eigvec = ElectronicIntegrals.apply(
            np.linalg.eigh, einsummed, multi=True, validate=False
        )

        mo_coeff_vir_new = ElectronicIntegrals.einsum(
            {"ij,jk->ik": ("+-",) * 3}, mo_coeff_vir_new, eigvec, validate=False
        )

        mo_coeff_vir_kern_alpha = mo_coeff_vir_kern.alpha
        if (
            "+-" in mo_coeff_vir_kern_alpha
            and mo_coeff_vir_kern_alpha["+-"].shape[1] != 0
        ):
            einsummed = PolynomialTensor.einsum(
                {"ji,jk,kl->il": ("+-",) * 4},
                mo_coeff_vir_kern_alpha,
                fock.alpha,
                mo_coeff_vir_kern_alpha,
                validate=False,
            )

            _, eigvec = np.linalg.eigh(einsummed["+-"])

            mo_coeff_vir_kern_alpha = PolynomialTensor.einsum(
                {"ij,jk->ik": ("+-",) * 3},
                mo_coeff_vir_kern_alpha,
                PolynomialTensor({"+-": eigvec}, validate=False),
                validate=False,
            )

            mo_coeff_vir_alpha = PolynomialTensor.stack(
                np.hstack,
                (mo_coeff_vir_new.alpha, mo_coeff_vir_kern_alpha),
                validate=False,
            )
        else:
            mo_coeff_vir_alpha = mo_coeff_vir_new.alpha

        mo_coeff_vir_kern_beta = mo_coeff_vir_kern.beta
        if (
            "+-" in mo_coeff_vir_kern_beta
            and mo_coeff_vir_kern_beta["+-"].shape[1] != 0
        ):
            einsummed = PolynomialTensor.einsum(
                {"ji,jk,kl->il": ("+-",) * 4},
                mo_coeff_vir_kern_beta,
                fock.beta,
                mo_coeff_vir_kern_beta,
                validate=False,
            )

            _, eigvec = np.linalg.eigh(einsummed["+-"])

            mo_coeff_vir_kern_beta = PolynomialTensor.einsum(
                {"ij,jk->ik": ("+-",) * 3},
                mo_coeff_vir_kern_beta,
                PolynomialTensor({"+-": eigvec}, validate=False),
                validate=False,
            )

            mo_coeff_vir_beta = PolynomialTensor.stack(
                np.hstack,
                (mo_coeff_vir_new.beta, mo_coeff_vir_kern_beta),
                validate=False,
            )
        else:
            mo_coeff_vir_beta = mo_coeff_vir_new.beta

        mo_coeff_vir = ElectronicIntegrals(mo_coeff_vir_alpha, mo_coeff_vir_beta)

        nvir_a_alpha = mo_coeff_vir_new.alpha["+-"].shape[1]
        nvir_b_alpha = mo_coeff_vir.alpha["+-"].shape[1] - nvir_a_alpha

        # WARNING: the number of alpha- and beta-spin orbitals is not guaranteed to be identical after
        # this step!
        nvir_a_beta, nvir_b_beta = None, None
        if "+-" in mo_coeff_vir_new.beta:
            nvir_a_beta = mo_coeff_vir_new.beta["+-"].shape[1]
            nvir_b_beta = mo_coeff_vir.beta["+-"].shape[1] - nvir_a_beta

        return mo_coeff_vir, (nvir_a_alpha, nvir_a_beta), (nvir_b_alpha, nvir_b_beta)
