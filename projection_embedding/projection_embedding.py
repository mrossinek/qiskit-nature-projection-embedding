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
        nmo = problem.num_spatial_orbitals

        nfc = self.num_frozen_occupied_orbitals

        # TODO: can we deal with unrestricted spin cases?
        mo_coeff = self.basis_transformer.coefficients.alpha["+-"]
        mo_coeff_occ = mo_coeff[:, :nocc]
        mo_coeff_unocc = mo_coeff[:, nocc:]

        fragment_1 = np.zeros((nao, nocc_a))
        fragment_2 = np.zeros((nao, nocc_b))

        overlap = problem.overlap_matrix
        overlap[np.abs(overlap) < 1e-12] = 0.0

        # TODO: this cannot be optional, right? Otherwise the fragments remain all-zero...
        # Maybe the idea was to allow exchanging SPADE with another localization scheme?
        if self.do_spade:
            fragment_1, fragment_2 = self._spade_partition(overlap, mo_coeff_occ, nocc_a)

        density_frozen = ElectronicDensity.from_raw_integrals(
            fragment_2.dot(fragment_2.transpose())
        )

        density_a = ElectronicDensity.from_raw_integrals(fragment_1.dot(fragment_1.transpose()))

        e_low_level = 0.0
        fock_, e_low_level = self._fock_build_a(density_a, density_frozen, hamiltonian)

        projector = np.identity(nao) - overlap.dot(density_frozen.alpha["+-"])
        fock = np.dot(projector, np.dot(fock_.alpha["+-"], projector.transpose()))

        e_old = 0
        e_thres = 1e-7
        max_iter = 50

        logger.info("")
        logger.info(" Hartree-Fock for subsystem A Energy")

        e_nuc = hamiltonian.nuclear_repulsion_energy

        for scf_iter in range(1, max_iter + 1):

            _, mo_coeff_a_full = la.eigh(fock, overlap)
            fragment_1 = mo_coeff_a_full[:, :nocc_a]

            density_a = ElectronicDensity.from_raw_integrals(fragment_1.dot(fragment_1.transpose()))

            fock_, e_low_level = self._fock_build_a(density_a, density_frozen, hamiltonian)

            projector = np.identity(nao) - overlap.dot(density_frozen.alpha["+-"])
            fock = np.dot(projector, np.dot(fock_.alpha["+-"], projector.transpose()))

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

        # Post iterations
        logger.info("\nSCF converged.")
        logger.info("Final SCF A-in-B Energy: %s [Eh]", e_new_a)

        # post convergence wrapup
        projector = np.dot(overlap, np.dot(density_frozen.alpha["+-"], overlap))

        fock_, e_low_level = self._fock_build_a(density_a, density_frozen, hamiltonian)

        fock = fock_.alpha["+-"]

        mu = 1.0e8
        fock -= mu * projector

        density_full = density_a + density_frozen
        mo_coeff_projected = np.dot(density_full.alpha["+-"], np.dot(overlap, mo_coeff_unocc))

        if np.linalg.norm(mo_coeff_projected) < 1e-05:
            logger.info("occupied and unoccupied are orthogonal")
            nonorthogonal = False
        else:
            logger.info("occupied and unoccupied are NOT orthogonal")
            nonorthogonal = True

        # orthogonalization procedure
        if nonorthogonal:
            mo_coeff_unocc_projected = mo_coeff_unocc - mo_coeff_projected

            eigval, eigvec = np.linalg.eigh(
                np.dot(
                    mo_coeff_unocc_projected.T, np.dot(overlap, mo_coeff_unocc_projected)
                )
            )

            eigval = np.linalg.inv(np.diag(np.sqrt(eigval)))

            mo_coeff_unocc = np.dot(mo_coeff_unocc_projected, np.dot(eigvec, eigval))

            _, eigvec_fock = np.linalg.eigh(
                np.dot(mo_coeff_unocc.T, np.dot(fock, mo_coeff_unocc))
            )
            mo_coeff_unocc = np.dot(mo_coeff_unocc, eigvec_fock)

        # doing concentric local virtuals
        zeta = 1
        nvir = nmo - nocc
        nvir_act = nvir
        nvir_frozen = 0
        mo_coeff_unocc_prime, nvir_act, nvir_frozen = self._get_truncated_virtuals(
            overlap, overlap, mo_coeff_unocc, fock, zeta
        )
        logger.debug("nvir_act = %s", nvir_act)
        logger.debug("nvir_frozen = %s", nvir_frozen)

        mo_coeff_excld_v = mo_coeff_unocc_prime[:, nvir_act:]
        proj_excluded_virts = np.dot(
            overlap, np.dot(mo_coeff_excld_v, np.dot(mo_coeff_excld_v.transpose(), overlap))
        )
        fock += mu * proj_excluded_virts

        mo_coeff_embedded_ncols = fragment_1.shape[1]
        mo_coeff_full_system_truncated = np.zeros(
            (fragment_1.shape[0], fragment_1.shape[1] + nvir_act)
        )
        mo_coeff_full_system_truncated[:, :mo_coeff_embedded_ncols] = fragment_1
        mo_coeff_full_system_truncated[:, mo_coeff_embedded_ncols:] = mo_coeff_unocc_prime[
            :, :nvir_act
        ]

        nmo_a_tmp = mo_coeff_full_system_truncated.shape[1] - self.num_frozen_virtual_orbitals
        mo_coeff_full_system_truncated = mo_coeff_full_system_truncated[:, nfc:nmo_a_tmp]
        nmo_a = mo_coeff_full_system_truncated.shape[1]
        logger.debug("nmo_a = %s", nmo_a)
        nocc_a -= nfc

        orbital_energy_mat = np.dot(
            mo_coeff_full_system_truncated.transpose(), np.dot(fock, mo_coeff_full_system_truncated)
        )
        orbital_energy = np.diag(orbital_energy_mat)
        logger.info(orbital_energy)
        logger.info("")
        logger.info("starting with the WFN-in-SCF calculation")

        ###Storing integrals to FCIDUMP

        transform = BasisTransformer(
            ElectronicBasis.AO,
            ElectronicBasis.MO,
            ElectronicIntegrals.from_raw_integrals(mo_coeff_full_system_truncated, validate=False),
        )

        new_hamiltonian = cast(ElectronicEnergy, transform.transform_hamiltonian(hamiltonian))

        only_a_mat = np.diag(
            [1.0 if i < nocc_a else 0.0 for i in range(mo_coeff_full_system_truncated.shape[1])]
        )
        only_a = ElectronicDensity.from_raw_integrals(only_a_mat)

        e_new_a_only = float(
            ElectronicIntegrals.einsum(
                {"ij,ji": ("+-", "+-", "")},
                new_hamiltonian.electronic_integrals.one_body + new_hamiltonian.fock(only_a),
                only_a,
            ).alpha[""]
        )

        logger.info("Final RHF A Energy        : %.14f [Eh]", e_new_a_only)
        logger.info("Final RHF A Energy tot    : %.14f [Eh]", e_new_a_only + e_nuc)

        new_hamiltonian.electronic_integrals -= new_hamiltonian.fock(only_a)
        new_hamiltonian.electronic_integrals += ElectronicIntegrals.from_raw_integrals(
            np.diag(orbital_energy)
        )

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
            print(f"e_mp2 = {e_mp2}")

        result = ElectronicStructureProblem(new_hamiltonian)
        result.num_particles = self.num_electrons - (self.num_frozen_occupied_orbitals * 2)
        result.num_spatial_orbitals = nmo_a

        return result

    def _fock_build_a(self, density_a, density_frozen, hamiltonian):
        density_tot = density_a + density_frozen

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

        # NOTE: the following is written as it is because this reflects better how DFT will differ
        fock_final = fock_a + (fock_tot - h_core) - (fock_a - h_core)

        return fock_final, e_low_level.alpha[""]

    def _get_truncated_virtuals(self, working_basis, projection_basis, mo_coeff_unocc, fock, zeta):

        logger.info("")
        logger.info("doing truncated local virtuals ")
        logger.info("    Concentric localization and truncation for virtuals    ")
        logger.info("     D. Claudino and N. Mayhall, JCTC, 15, 6085 (2019)     ")
        logger.info("")

        projection_basis_embed = projection_basis[
            : self.num_basis_functions, : self.num_basis_functions
        ]

        working_basis_red = working_basis[: self.num_basis_functions, :]

        projection_basis_embed_inv = np.linalg.inv(projection_basis_embed)
        mo_coeff_unocc_prime = np.dot(
            projection_basis_embed_inv, np.dot(working_basis_red, mo_coeff_unocc)
        )

        _, _, v_t = np.linalg.svd(
            np.dot(mo_coeff_unocc_prime.transpose(), np.dot(working_basis_red, mo_coeff_unocc)),
            full_matrices=True,
        )
        v = v_t.transpose()

        mo_coeff_unocc_new = np.dot(mo_coeff_unocc, v[:, : self.num_basis_functions])
        mo_coeff_unocc_rem = np.dot(mo_coeff_unocc, v[:, self.num_basis_functions :])

        for _ in range(zeta - 1):
            fock_new_term = np.dot(mo_coeff_unocc_new.transpose(), np.dot(fock, mo_coeff_unocc_rem))
            _, _, v_t = np.linalg.svd(fock_new_term, full_matrices=True)
            v = v_t.transpose()

            # update
            mo_coeff_unocc_rem_ncols = mo_coeff_unocc_rem.shape[1]
            mo_coeff_unocc_cur_ncols = mo_coeff_unocc_cur.shape[1]
            if mo_coeff_unocc_rem_ncols > mo_coeff_unocc_cur_ncols:
                mo_coeff_unocc_cur = np.dot(mo_coeff_unocc_rem, v[:, :mo_coeff_unocc_cur_ncols])
                mo_coeff_unocc_rem = np.dot(mo_coeff_unocc_rem, v[:, mo_coeff_unocc_cur_ncols:])

            else:
                mo_coeff_unocc_cur = np.dot(mo_coeff_unocc_rem, v)
                mo_coeff_unocc_rem = np.zeros(
                    (mo_coeff_unocc_rem.shape[0], mo_coeff_unocc_rem.shape[1])
                )

            mo_coeff_unocc_new_nrows = mo_coeff_unocc_new.shape[0]
            mo_coeff_unocc_new_ncols = mo_coeff_unocc_new.shape[1]
            mo_coeff_unocc_cur_ncols = mo_coeff_unocc_cur.shape[1]
            mo_coeff_unocc_new_tmp = np.zeros(
                (mo_coeff_unocc_new_nrows, mo_coeff_unocc_new_ncols + mo_coeff_unocc_cur_ncols)
            )
            mo_coeff_unocc_new_tmp[:, :mo_coeff_unocc_new_ncols] = mo_coeff_unocc_new
            mo_coeff_unocc_new_tmp[:, mo_coeff_unocc_new_ncols:] = mo_coeff_unocc_cur

            mo_coeff_unocc_new = mo_coeff_unocc_new_tmp

        logger.info("Pseudocanonicalizing the selected and excluded virtuals separately")

        fock_new_virtvirt = np.dot(mo_coeff_unocc_new.transpose(), np.dot(fock, mo_coeff_unocc_new))

        _, u_new = np.linalg.eigh(fock_new_virtvirt)

        mo_coeff_unocc_new = np.dot(mo_coeff_unocc_new, u_new)

        mo_coeff_unocc_rem_ncols = mo_coeff_unocc_rem.shape[1]
        if mo_coeff_unocc_rem_ncols != 0:
            fock_rem_virtvirt = np.dot(
                mo_coeff_unocc_rem.transpose(), np.dot(fock, mo_coeff_unocc_rem)
            )
            _, u_rem = np.linalg.eigh(fock_rem_virtvirt)

            mo_coeff_unocc_rem = np.dot(mo_coeff_unocc_rem, u_rem)
            mo_coeff_unocc_new_nrows = mo_coeff_unocc_new.shape[0]
            mo_coeff_unocc_new_ncols = mo_coeff_unocc_new.shape[1]
            mo_coeff_unocc_rem_ncols = mo_coeff_unocc_rem.shape[1]
            mo_coeff_unocc_tmp = np.zeros(
                (mo_coeff_unocc_new_nrows, mo_coeff_unocc_new_ncols + mo_coeff_unocc_rem_ncols)
            )
            mo_coeff_unocc_tmp[:, :mo_coeff_unocc_new_ncols] = mo_coeff_unocc_new
            mo_coeff_unocc_tmp[:, mo_coeff_unocc_new_ncols:] = mo_coeff_unocc_rem
            mo_coeff_unocc = mo_coeff_unocc_tmp
        else:
            mo_coeff_unocc = mo_coeff_unocc_new

        nvir_act = mo_coeff_unocc_new.shape[1]
        nvir_frozen = mo_coeff_unocc.shape[1] - nvir_act

        return mo_coeff_unocc, nvir_act, nvir_frozen

    def _spade_partition(self, overlap: np.ndarray, mo_coeff_occ: np.ndarray, nocc_a: int):
        logger.info("")
        logger.info("Doing SPADE partitioning")
        logger.info("D. CLaudino and N. Mayhall JCTC 15, 1053 (2019)")
        logger.info("")

        # 1. use symmetric orthogonalization on the overlap matrix
        symm_orth = symmetric_orthogonalization(overlap)

        # 2. change the MO basis to be orthogonal and reasonably localized
        mo_coeff_tmp = np.dot(symm_orth, mo_coeff_occ)

        # 3. select the active sector
        mo_coeff_tmp = mo_coeff_tmp[: self.num_basis_functions, :]

        # 4. use SVD to find the final rotation matrix
        _, _, rot_T = np.linalg.svd(mo_coeff_tmp, full_matrices=True)
        rot = rot_T.transpose()

        return np.dot(mo_coeff_occ, rot[:, :nocc_a]), np.dot(mo_coeff_occ, rot[:, nocc_a:])
