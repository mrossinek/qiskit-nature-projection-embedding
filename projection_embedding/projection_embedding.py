# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Projection Embedding transformer."""

from __future__ import annotations

from typing import cast

import logging

import numpy as np
import scipy.linalg as la

from qiskit_nature.second_q.hamiltonians import ElectronicEnergy, Hamiltonian
from qiskit_nature.second_q.operators import ElectronicIntegrals
from qiskit_nature.second_q.problems import BaseProblem, ElectronicBasis, ElectronicStructureProblem
from qiskit_nature.second_q.properties import ElectronicDensity
from qiskit_nature.utils import symmetric_orthogonalization

from .base_transformer import BaseTransformer
from .basis_transformer import BasisTransformer

logger = logging.getLogger(__name__)


class ProjectionTransformer(BaseTransformer):
    """TODO."""

    def __init__(
        self,
        num_electrons: int,  # the number of electrons in the "active" subsystem A
        num_basis_functions: int,  # the number of basis functions in the "active" subsystem A
        num_frozen_occupied_orbitals: int,  # the number of occupied orbitals to freeze
        num_frozen_virtual_orbitals: int,  # the number of virtual orbitals to freeze
        basis_transformer: BasisTransformer,
        *,
        do_spade: bool = True,
    ) -> None:
        """TODO."""
        self.num_electrons = num_electrons
        self.num_basis_functions = num_basis_functions
        self.num_frozen_occupied_orbitals = num_frozen_occupied_orbitals
        self.num_frozen_virtual_orbitals = num_frozen_virtual_orbitals
        self.basis_transformer = basis_transformer
        self.do_spade = do_spade

    def transform(self, problem: BaseProblem) -> BaseProblem:
        """TODO."""
        if isinstance(problem, ElectronicStructureProblem):
            return self._transform_electronic_structure_problem(problem)
        else:
            raise NotImplementedError(
                f"The problem of type, {type(problem)}, is not supported by this transformer."
            )

    def transform_hamiltonian(self, hamiltonian: Hamiltonian) -> Hamiltonian:
        """TODO."""
        if isinstance(hamiltonian, ElectronicEnergy):
            pass
        else:
            raise NotImplementedError(
                f"The hamiltonian of type, {type(hamiltonian)}, is not supported by this "
                "transformer."
            )

    def _transform_electronic_structure_problem(
        self, problem: ElectronicStructureProblem
    ) -> ElectronicStructureProblem:
        logger.info("")
        logger.info("Starting with the Manby-Miller embedding")
        logger.info("F. Manby et al. JCTC, 8, 2564 (2012) ")
        logger.info("")
        logger.info("")
        logger.info("Starting with embedding calculation")
        logger.info("Doing SCF-in-SCF embedding calculation")
        logger.info("")

        # TODO: assert AO basis

        hamiltonian = problem.hamiltonian

        num_elec_env = sum(problem.num_particles) - self.num_electrons
        nao = self.basis_transformer.coefficients.alpha["+-"].shape[0]
        nocc_a = self.num_electrons // 2
        nocc_b = num_elec_env // 2
        nocc = nocc_a + nocc_b

        # TODO: can we deal with unrestricted spin cases?
        mo_coeff = self.basis_transformer.coefficients.alpha["+-"]
        mo_coeff_occ, mo_coeff_vir = np.hsplit(mo_coeff, [nocc])

        fragment_a = np.zeros((nao, nocc_a))
        fragment_b = np.zeros((nao, nocc_b))

        overlap = problem.overlap_matrix
        overlap[np.abs(overlap) < 1e-12] = 0.0
        overlap_ints = ElectronicIntegrals.from_raw_integrals(overlap)

        # TODO: this cannot be optional, right? Otherwise the fragments remain all-zero...
        # Maybe the idea was to allow exchanging SPADE with another localization scheme?
        # TODO: make standalone orbital localization method which can be overwritten by the user
        if self.do_spade:
            fragment_a, fragment_b = _spade_partition(
                overlap, mo_coeff_occ, self.num_basis_functions, nocc_a
            )

        # NOTE: fragment_a will ONLY change if the SCF loop below is necessary to ensure consistent
        # embedding (which I believe to be trivial in the HF case and, thus, only occur with DFT)

        density_b = ElectronicDensity.from_raw_integrals(fragment_b.dot(fragment_b.T))

        density_a = ElectronicDensity.from_raw_integrals(fragment_a.dot(fragment_a.T))

        fock_, e_low_level = _fock_build_a(density_a, density_b, hamiltonian)

        identity = ElectronicIntegrals.from_raw_integrals(np.identity(nao))
        projector = identity - ElectronicIntegrals.einsum(
            {"ij,jk->ik": ("+-",) * 3}, overlap_ints, density_b
        )
        fock = ElectronicIntegrals.einsum(
            {"ij,jk,lk->il": ("+-",) * 4}, projector, fock_, projector
        )

        e_old = 0
        e_thres = 1e-7
        max_iter = 50

        logger.info("")
        logger.info(" Hartree-Fock for subsystem A Energy")

        e_nuc = hamiltonian.nuclear_repulsion_energy

        # TODO: is this SCF loop necessary in the HF case?
        for scf_iter in range(1, max_iter + 1):

            _, mo_coeff_a_full = la.eigh(fock.alpha["+-"], overlap)
            fragment_a = mo_coeff_a_full[:, :nocc_a]

            density_a = ElectronicDensity.from_raw_integrals(fragment_a.dot(fragment_a.T))

            fock_, e_low_level = _fock_build_a(density_a, density_b, hamiltonian)

            projector = identity - ElectronicIntegrals.einsum(
                {"ij,jk->ik": ("+-",) * 3}, overlap_ints, density_b
            )
            fock = ElectronicIntegrals.einsum(
                {"ij,jk,lk->il": ("+-",) * 4}, projector, fock_, projector
            )

            e_new_a = ElectronicIntegrals.einsum(
                {"ij,ji": ("+-", "+-", "")},
                hamiltonian.electronic_integrals.one_body + hamiltonian.fock(density_a),
                density_a,
            ).alpha[""]

            e_new_a += e_low_level + e_nuc

            logger.info("SCF Iteration %s: Energy = %s dE = %s", scf_iter, e_new_a, e_new_a - e_old)

            # SCF Converged?
            if abs(e_new_a - e_old) < e_thres:
                break
            e_old = e_new_a

            if scf_iter == max_iter:
                raise Exception("Maximum number of SCF iterations exceeded.")

        # NOTE: from now on fragment_a will no longer change!

        # Post iterations
        logger.info("\nSCF converged.")
        logger.info("Final SCF A-in-B Energy: %s [Eh]", e_new_a)

        # post convergence wrapup
        fock_, e_low_level = _fock_build_a(density_a, density_b, hamiltonian)

        fock = fock_.alpha["+-"]

        mu = 1.0e8
        # TODO: what exactly is this step in which we subtract a projector?
        # am I write assuming that this deals with the occupied orbitals of subsystem B?
        fock -= mu * np.einsum(
            "ij,jk,kl->il", overlap, density_b.alpha["+-"], overlap, optimize=True
        )

        density_full = density_a + density_b
        mo_coeff_projected = np.einsum(
            "ij,jk,kl->il", density_full.alpha["+-"], overlap, mo_coeff_vir, optimize=True
        )

        if np.linalg.norm(mo_coeff_projected) < 1e-05:
            logger.info("occupied and unoccupied are orthogonal")
            nonorthogonal = False
        else:
            logger.info("occupied and unoccupied are NOT orthogonal")
            nonorthogonal = True

        # orthogonalization procedure
        # TODO: which procedure is this exactly and when does it become necessary? In the DFT case?
        # indeed, this is not triggered in the HF-case because the orbitals are already orthogonal
        # (at least for systems known to us)
        if nonorthogonal:
            mo_coeff_vir_projected = mo_coeff_vir - mo_coeff_projected

            eigval, eigvec = np.linalg.eigh(
                np.einsum(
                    "ji,jk,kl->il",
                    mo_coeff_vir_projected,
                    overlap,
                    mo_coeff_vir_projected,
                    optimize=True,
                )
            )

            eigval = np.linalg.inv(np.diag(np.sqrt(eigval)))

            mo_coeff_vir = np.einsum(
                "ij,jk,kl->il", mo_coeff_vir_projected, eigvec, eigval, optimize=True
            )

            _, eigvec_fock = np.linalg.eigh(
                np.einsum("ji,jk,kl->il", mo_coeff_vir, fock, mo_coeff_vir, optimize=True)
            )
            mo_coeff_vir = np.dot(mo_coeff_vir, eigvec_fock)

        # doing concentric local virtuals
        mo_coeff_vir_pb, nvir_a, nvir_b = _concentric_localization(
            overlap[: self.num_basis_functions, :],
            overlap[: self.num_basis_functions, : self.num_basis_functions],
            mo_coeff_vir,
            self.num_basis_functions,
            fock,
            zeta=1,  # TODO: make configurable and figure out what exactly zeta is meant to do?
        )
        logger.debug("nvir_a = %s", nvir_a)
        logger.debug("nvir_b = %s", nvir_b)

        mo_coeff_vir_a, mo_coeff_vir_b = np.hsplit(mo_coeff_vir_pb, [nvir_a])

        # P^B in Manby2012
        proj_excluded_virts = np.einsum(
            "ij,jk,lk,lm->im", overlap, mo_coeff_vir_b, mo_coeff_vir_b, overlap, optimize=True
        )
        fock += mu * proj_excluded_virts

        max_orb = -self.num_frozen_virtual_orbitals if self.num_frozen_virtual_orbitals else None
        mo_coeff_final = np.hstack((fragment_a, mo_coeff_vir_a))[
            :, self.num_frozen_occupied_orbitals : max_orb
        ]
        logger.debug("nmo_a = %s", mo_coeff_final.shape[1])
        nocc_a -= self.num_frozen_occupied_orbitals

        # NOTE: at this point fock =  (omitting pre-factors for Coulomb to be RKS/UKS-agnostic)
        #   h_core + J_A - K_A
        #   + J_tot - xc_fac * K_tot + Vxc_tot
        #   - (J_A - xc_fac * K_A + Vxc_A)
        #   - mu * P_B_occ
        #   + mu * P_B_vir
        # NOTE: this matches Eq. (3) from Many2012 with the additional J_A and K_A terms;
        # these are included to take the 2-body terms into account
        orbital_energy_mat = np.einsum(
            "ji,jk,kl->il", mo_coeff_final, fock, mo_coeff_final, optimize=True
        )
        orbital_energy = np.diag(orbital_energy_mat)
        logger.info("orbital energies")
        logger.info(orbital_energy)

        transform = BasisTransformer(
            ElectronicBasis.AO,
            ElectronicBasis.MO,
            ElectronicIntegrals.from_raw_integrals(mo_coeff_final, validate=False),
        )

        new_hamiltonian = cast(ElectronicEnergy, transform.transform_hamiltonian(hamiltonian))
        # now, new_hamiltonian is simply the hamiltonian we started with but transformed into the
        # projected MO basis (which is limited to subsystem A)

        only_a = ElectronicDensity.from_raw_integrals(
            np.diag([1.0 if i < nocc_a else 0.0 for i in range(mo_coeff_final.shape[1])])
        )

        e_new_a_only = float(
            ElectronicIntegrals.einsum(
                {"ij,ji": ("+-", "+-", "")},
                new_hamiltonian.electronic_integrals.one_body + new_hamiltonian.fock(only_a),
                only_a,
            ).alpha[""]
        )

        logger.info("Final RHF A Energy        : %.14f [Eh]", e_new_a_only)
        logger.info("Final RHF A Energy tot    : %.14f [Eh]", e_new_a_only + e_nuc)

        # next, we subtract the fock operator from the hamiltonian which subtracts h + J_A - K_A
        new_hamiltonian.electronic_integrals -= new_hamiltonian.fock(only_a)
        # and finally we add the diagonal matrix containing the orbital energies
        new_hamiltonian.electronic_integrals += ElectronicIntegrals.from_raw_integrals(
            np.diag(orbital_energy)
        )
        # TODO: to verify: this re-adds the contributions from h + J_tot - K_tot + \mu * P_B
        # this is a trick to avoid double counting
        # new_hamiltonian now equals h_{A in B} (see Eq. (3) in Manby2012)

        e_new_a_only = ElectronicIntegrals.einsum(
            {"ij,ji": ("+-", "+-", "")},
            new_hamiltonian.electronic_integrals.one_body + new_hamiltonian.fock(only_a),
            only_a,
        ).alpha[""]

        logger.info("Final RHF A eff Energy        : %.14f [Eh]", e_new_a_only)
        logger.info("Final RHF A eff Energy tot    : %.14f [Eh]", e_new_a_only + e_nuc)

        new_hamiltonian.nuclear_repulsion_energy = float(e_new_a)
        new_hamiltonian.constants["ProjectionTransformer"] = -1.0 * float(e_new_a_only)

        if logger.isEnabledFor(logging.INFO):

            from qiskit_nature.second_q.algorithms.initial_points.mp2_initial_point import (
                _compute_mp2,
            )

            _, e_mp2 = _compute_mp2(
                nocc_a, new_hamiltonian.electronic_integrals.two_body.alpha["++--"], orbital_energy
            )

            logger.info("e_mp2 = %4.10f", e_mp2)

        result = ElectronicStructureProblem(new_hamiltonian)
        result.num_particles = self.num_electrons - (self.num_frozen_occupied_orbitals * 2)
        result.num_spatial_orbitals = mo_coeff_final.shape[1]

        return result


def _fock_build_a(density_a, density_b, hamiltonian):
    density_tot = density_a + density_b

    # NOTE: in the DFT case, these need to include the XC components
    fock_a = hamiltonian.fock(density_a)
    fock_tot = hamiltonian.fock(density_tot)

    h_core = hamiltonian.electronic_integrals.one_body

    e_low_level = ElectronicIntegrals.einsum(
        {"ij,ji": ("+-", "+-", "")},
        fock_tot + h_core,
        density_tot,
    )
    e_low_level -= ElectronicIntegrals.einsum(
        {"ij,ji": ("+-", "+-", "")},
        fock_a + h_core,
        density_a,
    )
    # TODO: in the DFT case we need to additionally deal with the XC components
    # we can handle this via an optional callback

    # NOTE: the following is written as it is because this reflects better how DFT will differ
    fock_final = hamiltonian.fock(density_a)  # NOTE: this should NOT contain any XC components
    fock_final += (fock_tot - h_core) - (fock_a - h_core)

    return fock_final, e_low_level.alpha[""]


def _concentric_localization(overlap_pb_wb, projection_basis, mo_coeff_vir, num_bf, fock, zeta):
    logger.info("")
    logger.info("Doing concentric location and truncation of virtual space")
    logger.info("    Concentric localization and truncation for virtuals    ")
    logger.info("     D. Claudino and N. Mayhall, JCTC, 15, 6085 (2019)     ")
    logger.info("")

    # S^{-1} in paper
    overlap_a_pb_inv = np.linalg.inv(projection_basis)

    # C'_{vir} in paper
    mo_coeff_vir_pb = np.einsum(
        "ij,jk,kl->il", overlap_a_pb_inv, overlap_pb_wb, mo_coeff_vir, optimize=True
    )

    # Eq. (10a)
    _, _, v_t = np.linalg.svd(
        np.einsum("ji,jk,kl->il", mo_coeff_vir_pb, overlap_pb_wb, mo_coeff_vir, optimize=True),
        full_matrices=True,
    )
    # Eq. (10b)
    v_span, v_kern = np.hsplit(v_t.T, [num_bf])

    # Eq. (10c)
    mo_coeff_vir_new = np.dot(mo_coeff_vir, v_span)

    # Eq. (10d)
    mo_coeff_vir_kern = np.dot(mo_coeff_vir, v_kern)

    # zeta controls the number of unoccupied orbitals
    for _ in range(zeta - 1):
        # Eq. (12a)
        _, _, v_t = np.linalg.svd(
            np.einsum("ji,jk,kl->il", mo_coeff_vir_new, fock, mo_coeff_vir_kern, optimize=True),
            full_matrices=True,
        )

        # Eq. (12b)
        v_span, v_kern = np.hsplit(v_t.T, [mo_coeff_vir_cur.shape[1]])

        # Eq. (12c-12d)
        if mo_coeff_vir_kern.shape[1] > mo_coeff_vir_cur.shape[1]:
            mo_coeff_vir_cur = np.dot(mo_coeff_vir_kern, v_span)
            mo_coeff_vir_kern = np.dot(mo_coeff_vir_kern, v_kern)

        else:
            mo_coeff_vir_cur = np.dot(mo_coeff_vir_kern, v_t.T)
            mo_coeff_vir_kern = np.zeros_like(mo_coeff_vir_kern)

        # Eq. (12e)
        mo_coeff_vir_new = np.hstack((mo_coeff_vir_new, mo_coeff_vir_cur))

    # post-processing step
    logger.info("Pseudocanonicalizing the selected and excluded virtuals separately")

    _, eigvecs = np.linalg.eigh(
        np.einsum("ji,jk,kl->il", mo_coeff_vir_new, fock, mo_coeff_vir_new, optimize=True)
    )

    mo_coeff_vir_new = np.dot(mo_coeff_vir_new, eigvecs)

    if mo_coeff_vir_kern.shape[1] != 0:
        _, eigvecs = np.linalg.eigh(
            np.einsum("ji,jk,kl->il", mo_coeff_vir_kern, fock, mo_coeff_vir_kern, optimize=True)
        )

        mo_coeff_vir_kern = np.dot(mo_coeff_vir_kern, eigvecs)

        mo_coeff_vir = np.hstack((mo_coeff_vir_new, mo_coeff_vir_kern))
    else:
        mo_coeff_vir = mo_coeff_vir_new

    nvir_a = mo_coeff_vir_new.shape[1]
    nvir_b = mo_coeff_vir.shape[1] - nvir_a

    return mo_coeff_vir, nvir_a, nvir_b


def _spade_partition(overlap: np.ndarray, mo_coeff_occ: np.ndarray, num_bf: int, nocc_a: int):
    logger.info("")
    logger.info("Doing SPADE partitioning")
    logger.info("D. CLaudino and N. Mayhall JCTC 15, 1053 (2019)")
    logger.info("")

    # 1. use symmetric orthogonalization on the overlap matrix
    symm_orth = symmetric_orthogonalization(overlap)

    # 2. change the MO basis to be orthogonal and reasonably localized
    mo_coeff_tmp = np.dot(symm_orth, mo_coeff_occ)

    # 3. select the active sector
    mo_coeff_tmp = mo_coeff_tmp[:num_bf, :]

    # 4. use SVD to find the final rotation matrix
    _, _, rot = np.linalg.svd(mo_coeff_tmp, full_matrices=True)
    left, right = np.hsplit(rot.T, [nocc_a])

    return np.dot(mo_coeff_occ, left), np.dot(mo_coeff_occ, right)
