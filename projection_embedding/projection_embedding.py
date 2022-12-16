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

import logging

import numpy as np
import scipy.linalg as la

from qiskit_nature.second_q.hamiltonians import ElectronicEnergy, Hamiltonian
from qiskit_nature.second_q.operators import PolynomialTensor
from qiskit_nature.second_q.operators.tensor_ordering import to_chemist_ordering
from qiskit_nature.second_q.problems import BaseProblem, ElectronicStructureProblem
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

        mo_coeff_occ_frozen = fragment_2
        density_frozen = mo_coeff_occ_frozen.dot(mo_coeff_occ_frozen.transpose())
        mo_coeff_occ_embedded = fragment_1

        mo_coeff_full_system = np.zeros((nao, nmo))
        mo_coeff_full_system[:, :nocc_a] = mo_coeff_occ_embedded
        mo_coeff_full_system[:, nocc_a:nocc] = mo_coeff_occ_frozen
        mo_coeff_full_system[:, nocc:] = mo_coeff_unocc

        h_core = hamiltonian.electronic_integrals.alpha["+-"]
        g_ao = to_chemist_ordering(hamiltonian.electronic_integrals.alpha["++--"])

        e_low_level = 0.0
        fock, e_low_level = self._fock_build_A(
            True, mo_coeff_full_system, density_frozen, nao, nocc_a, overlap, h_core, g_ao
        )

        density_a = mo_coeff_occ_embedded.dot(mo_coeff_occ_embedded.transpose())

        e_old = 0
        e_thres = 1e-7
        max_iter = 50

        logger.info("")
        logger.info(" Hartree-Fock for subsystem A Energy")

        e_nuc = hamiltonian.nuclear_repulsion_energy

        for scf_iter in range(1, max_iter + 1):

            _, mo_coeff_a_full = la.eigh(fock, overlap)
            mo_coeff_occ_embedded = mo_coeff_a_full[:, :nocc_a]

            density_a = mo_coeff_occ_embedded.dot(mo_coeff_occ_embedded.transpose())

            mo_coeff_full_system[:, :nocc_a] = mo_coeff_occ_embedded
            mo_coeff_full_system[:, nocc_a:nocc] = mo_coeff_occ_frozen
            mo_coeff_full_system[:, nocc:] = mo_coeff_unocc

            fock, e_low_level = self._fock_build_A(
                True, mo_coeff_full_system, density_frozen, nao, nocc_a, overlap, h_core, g_ao
            )

            e_new_a = np.einsum("pq,pq->", 2 * h_core, density_a, optimize=True)
            e_new_a += np.einsum("pq,pqrs,rs->", 2 * density_a, g_ao, density_a, optimize=True)
            e_new_a -= np.einsum("pq,prqs,rs->", 1 * density_a, g_ao, density_a, optimize=True)

            e_new_a += e_low_level + e_nuc

            logger.info(f"SCF Iteration {scf_iter}: Energy = {e_new_a} dE = {e_new_a - e_old}")

            # SCF Converged?
            if abs(e_new_a - e_old) < e_thres:
                break
            e_old = e_new_a

            if scf_iter == max_iter:
                raise Exception("Maximum number of SCF iterations exceeded.")

        # Post iterations
        logger.info("\nSCF converged.")
        logger.info(f"Final SCF A-in-B Energy: {e_new_a} [Eh]")

        # post convergence wrapup
        projector = np.dot(overlap, np.dot(density_frozen, overlap))

        fock, e_low_level = self._fock_build_A(
            False, mo_coeff_full_system, density_frozen, nao, nocc_a, overlap, h_core, g_ao
        )

        mu = 1.0e8
        fock -= mu * projector

        density_full = density_a + density_frozen
        mo_coeff_projected = np.dot(density_full, np.dot(overlap, mo_coeff_unocc))

        if np.linalg.norm(mo_coeff_projected) < 1e-05:
            logger.info("occupied and unoccupied are orthogonal")
            nonorthogonal = False
        else:
            logger.info("occupied and unoccupied are NOT orthogonal")
            nonorthogonal = True

        # orthogonalization procedure
        if nonorthogonal == True:
            mo_coeff_unocc_projected = mo_coeff_unocc - mo_coeff_projected

            eval, evec = np.linalg.eigh(
                np.dot(
                    mo_coeff_unocc_projected.transpose(), np.dot(overlap, mo_coeff_unocc_projected)
                )
            )

            for i in range(evec.shape[0]):
                eval[i] = eval[i] ** (-0.5)
            eval = np.diag(eval)

            mo_coeff_unocc = np.dot(mo_coeff_unocc_projected, np.dot(evec, eval))

            _, evec_fock = np.linalg.eigh(
                np.dot(mo_coeff_unocc.transpose(), np.dot(fock, mo_coeff_unocc))
            )
            mo_coeff_unocc = np.dot(mo_coeff_unocc, evec_fock)

        # doing concentric local virtuals
        zeta = 1
        nvir = nmo - nocc
        nvir_act = nvir
        nvir_frozen = 0
        mo_coeff_unocc_prime, nvir_act, nvir_frozen = self._get_truncated_virtuals(
            overlap, overlap, mo_coeff_unocc, fock, zeta
        )
        logger.debug(f"nvir_act = {nvir_act}")
        logger.debug(f"nvir_frozen = {nvir_frozen}")

        C_excld_v = mo_coeff_unocc_prime[:, nvir_act:]
        proj_excluded_virts = np.dot(
            overlap, np.dot(C_excld_v, np.dot(C_excld_v.transpose(), overlap))
        )
        fock += mu * proj_excluded_virts

        mo_coeff_embedded_ncols = mo_coeff_occ_embedded.shape[1]
        mo_coeff_full_system_truncated = np.zeros(
            (mo_coeff_occ_embedded.shape[0], mo_coeff_occ_embedded.shape[1] + nvir_act)
        )
        mo_coeff_full_system_truncated[:, :mo_coeff_embedded_ncols] = mo_coeff_occ_embedded
        mo_coeff_full_system_truncated[:, mo_coeff_embedded_ncols:] = mo_coeff_unocc_prime[
            :, :nvir_act
        ]

        nmo_a_tmp = mo_coeff_full_system_truncated.shape[1] - self.num_frozen_virtual_orbitals
        mo_coeff_full_system_truncated = mo_coeff_full_system_truncated[:, nfc:nmo_a_tmp]
        nmo_a = mo_coeff_full_system_truncated.shape[1]
        logger.debug(f"nmo_a = {nmo_a}")
        nocc_a -= nfc

        orbital_energy_mat = np.dot(
            mo_coeff_full_system_truncated.transpose(), np.dot(fock, mo_coeff_full_system_truncated)
        )
        orbital_energy = np.diag(orbital_energy_mat)
        logger.info(orbital_energy)
        logger.info("")
        logger.info("starting with the WFN-in-SCF calculation")

        ###Storing integrals to FCIDUMP

        g_mo_sf = PolynomialTensor.einsum(
            {"prsq,pi,qj,rk,sl->iklj": ("++--", *("+-",) * 4, "++--")},
            PolynomialTensor({"++--": g_ao}),
            *(PolynomialTensor({"+-": mo_coeff_full_system_truncated}, validate=False),) * 4,
        )["++--"]

        # AO2MO of one-body
        h_mo_sf = np.einsum(
            "pi,pq,qj -> ij",
            mo_coeff_full_system_truncated,
            h_core,
            mo_coeff_full_system_truncated,
            optimize=True,
        )

        # f = h + 2J-K
        f = np.diag(orbital_energy)
        h_eff = np.zeros((f.shape[0], f.shape[1]))
        h_eff += f
        h_eff -= np.einsum("pqii -> pq", 2.0 * g_mo_sf[:, :, :nocc_a, :nocc_a], optimize=True)
        h_eff += np.einsum("piqi -> pq", g_mo_sf[:, :nocc_a, :, :nocc_a], optimize=True)

        e_new_a_only = np.einsum("ii->", 2 * h_mo_sf[:nocc_a, :nocc_a], optimize=True)
        e_new_a_only += np.einsum(
            "iijj->", 2 * g_mo_sf[:nocc_a, :nocc_a, :nocc_a, :nocc_a], optimize=True
        )
        e_new_a_only -= np.einsum(
            "ijij->", 1 * g_mo_sf[:nocc_a, :nocc_a, :nocc_a, :nocc_a], optimize=True
        )
        logger.info("Final RHF A Energy        : %.14f [Eh]" % (e_new_a_only))
        e_new_a_only += e_nuc
        logger.info("Final RHF A Energy tot    : %.14f [Eh]" % (e_new_a_only))

        e_new_a_only = np.einsum("ii->", 2 * h_eff[:nocc_a, :nocc_a], optimize=True)
        e_new_a_only += np.einsum(
            "iijj->", 2 * g_mo_sf[:nocc_a, :nocc_a, :nocc_a, :nocc_a], optimize=True
        )
        e_new_a_only -= np.einsum(
            "ijij->", 1 * g_mo_sf[:nocc_a, :nocc_a, :nocc_a, :nocc_a], optimize=True
        )
        logger.info("Final RHF A eff Energy        : %.14f [Eh]" % (e_new_a_only))
        shift = -1.0 * float(e_new_a_only)
        e_new_a_only += e_nuc
        logger.info("Final RHF A eff Energy tot    : %.14f [Eh]" % (e_new_a_only))

        new_hamiltonian = ElectronicEnergy.from_raw_integrals(h_eff, g_mo_sf)
        new_hamiltonian.nuclear_repulsion_energy = float(e_new_a)
        new_hamiltonian.constants["ProjectionTransformer"] = shift

        result = ElectronicStructureProblem(new_hamiltonian)
        result.num_particles = self.num_electrons - (self.num_frozen_occupied_orbitals * 2)
        result.num_spatial_orbitals = nmo_a

        g_mo_sf = g_mo_sf.transpose(0, 2, 1, 3)

        e_ij = orbital_energy[:nocc_a]
        e_ab = orbital_energy[nocc_a:]

        # -1 means that it is an unknown dimension and we want numpy to figure it out.
        e_denom = 1 / (
            e_ij.reshape(-1, 1, 1, 1) - e_ab.reshape(-1, 1, 1) + e_ij.reshape(-1, 1) - e_ab
        )

        g_vvoo_ee = g_mo_sf[nocc_a:, nocc_a:, :nocc_a, :nocc_a]
        g_oovv_ee = g_mo_sf[:nocc_a, :nocc_a, nocc_a:, nocc_a:]
        t2_ee = np.einsum("iajb,abij->abij", e_denom, g_vvoo_ee, optimize=True)

        e_mp2 = np.einsum("ikac,acik->", g_oovv_ee, t2_ee, optimize=True)
        # +g_ee(ijab) * t2_ee(abij) * 0.5
        e_mp2 += np.einsum("ijab,abij->", 0.5 * g_oovv_ee, t2_ee, optimize=True)
        # -g_ee(ijab) * t2_ee(abji) * 0.5
        e_mp2 -= np.einsum("ijab,abji->", 0.5 * g_oovv_ee, t2_ee, optimize=True)
        # +g_ee(klcd) * t2_ee(cdkl) * 0.5
        e_mp2 += np.einsum("klcd,cdkl->", 0.5 * g_oovv_ee, t2_ee, optimize=True)
        # -g_ee(klcd) * t2_ee(cdlk) * 0.5
        e_mp2 -= np.einsum("klcd,cdlk->", 0.5 * g_oovv_ee, t2_ee, optimize=True)

        logger.info("e_mp2 = %4.10f" % (e_mp2))

        return result

    def _fock_build_A(
        self, project, mo_coeff_full_system, density_frozen, nao, nocc_a, overlap, h_core, g_ao
    ):
        # construction of the Fock matrix
        # Fragment A contribution
        mo_coeff_a = np.zeros((nao, nao))
        mo_coeff_a[:, :nocc_a] = mo_coeff_full_system[:, :nocc_a]

        density_a = mo_coeff_a.dot(mo_coeff_a.transpose())
        coulomb_a = np.einsum("pqrs,rs->pq", g_ao, density_a, optimize=True)
        exchange_a = np.einsum("prqs,rs->pq", g_ao, density_a, optimize=True)
        fock = h_core + 2 * coulomb_a - exchange_a

        density_full = density_a + density_frozen
        coulomb_full = np.einsum("pqrs,rs->pq", g_ao, density_full, optimize=True)
        exchange_full = np.einsum("prqs,rs->pq", g_ao, density_full, optimize=True)
        fock_tot_low_level = h_core + 2 * coulomb_full - exchange_full

        fock_a_low_level = h_core + 2 * coulomb_a - exchange_a

        fock += 2 * coulomb_full - exchange_full - (2 * coulomb_a - exchange_a)

        e_low_level = np.einsum(
            "pq,pq->", (h_core + fock_tot_low_level), density_full, optimize=True
        )
        e_low_level -= np.einsum("pq,pq->", (h_core + fock_a_low_level), density_a, optimize=True)

        if project == True:
            projector = np.identity(nao) - overlap.dot(density_frozen)
            fock = np.dot(projector, np.dot(fock, projector.transpose()))

        return fock, e_low_level

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
