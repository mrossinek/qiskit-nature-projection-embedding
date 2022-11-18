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

import numpy as np
import scipy.linalg as la

from qiskit_nature.second_q.hamiltonians import ElectronicEnergy, Hamiltonian
from qiskit_nature.second_q.operators import PolynomialTensor
from qiskit_nature.second_q.operators.tensor_ordering import to_chemist_ordering
from qiskit_nature.second_q.problems import BaseProblem, ElectronicBasis, ElectronicStructureProblem

from .base_transformer import BaseTransformer
from .basis_transformer import BasisTransformer


class ProjectionTransformer(BaseTransformer):
    """TODO."""

    def __init__(
        self,
        num_elec_A: int,
        num_bf_A: int,
        num_fc_elec: int,
        nfv: int,
        basis_transformer: BasisTransformer,
    ) -> None:
        """TODO."""
        self.num_elec_A = num_elec_A
        self.num_bf_A = num_bf_A
        self.num_fc_elec = num_fc_elec
        self.nfv = nfv
        self.basis_transformer = basis_transformer

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
        hamiltonian = problem.hamiltonian

        num_elec_B = sum(problem.num_particles) - self.num_elec_A
        num_bf_total = self.basis_transformer.coefficients.alpha["+-"].shape[0]
        num_bf_B = num_bf_total - self.num_bf_A

        print("")
        print("Starting with the Manby-Miller embedding")
        print("F. Manby et al. JCTC, 8, 2564 (2012) ")
        print("")
        print("")
        print("Starting with embedding calculation")
        print("Doing SCF-in-SCF embedding calculation")
        print("")

        # SPADE partition
        do_spade = True

        nocc_A = self.num_elec_A // 2
        nocc_B = num_elec_B // 2
        nocc = nocc_A + nocc_B
        nao = num_bf_total
        nmo = problem.num_spatial_orbitals

        num_fc_elec = self.num_fc_elec
        nfc = num_fc_elec // 2
        nfv = self.nfv

        C = self.basis_transformer.coefficients.alpha["+-"]
        C_occ = C[:, :nocc]
        C_uocc = C[:, nocc:]

        Frgment_LMO_1 = np.zeros((num_bf_total, nocc_A))
        Frgment_LMO_2 = np.zeros((num_bf_total, nocc_B))

        N_fragments = 2
        S = problem.overlap_matrix
        S[np.abs(S) < 1e-12] = 0.0
        if do_spade == True:
            # Doing SPADE partitioning
            Frgment_LMO_1, Frgment_LMO_2 = self._spade_partition(
                S, C_occ, N_fragments, nocc, nocc_A, self.num_bf_A
            )

        C_occ_frozen = Frgment_LMO_2
        P_frozen = C_occ_frozen.dot(C_occ_frozen.transpose())
        C_occ_embedded = Frgment_LMO_1

        C_full_system = np.zeros((num_bf_total, nmo))
        C_full_system[:, :nocc_A] = C_occ_embedded
        C_full_system[:, nocc_A:nocc] = C_occ_frozen
        C_full_system[:, nocc:] = C_uocc

        H_core = hamiltonian.electronic_integrals.alpha["+-"]
        g_ao = to_chemist_ordering(hamiltonian.electronic_integrals.alpha["++--"])

        E_low_level_of_theory = 0.0
        Fock, E_low_level_of_theory = self._fock_build_A(
            True, C_full_system, P_frozen, nao, nocc_A, S, H_core, g_ao
        )

        P_of_A = C_occ_embedded.dot(C_occ_embedded.transpose())
        P_of_A_Guess = P_of_A

        diff_SCF_E = 1
        E_old = 0
        E_threshold = 1e-7
        MAXITER = 50

        print("")
        print(" Hartree-Fock for subsystem A Energy")

        E_nuc = hamiltonian.nuclear_repulsion_energy

        for scf_iter in range(1, MAXITER + 1):

            energies, C_of_A_full = la.eigh(Fock, S)
            C_occ_embedded = C_of_A_full[:, :nocc_A]

            P_of_A = C_occ_embedded.dot(C_occ_embedded.transpose())

            C_full_system[:, :nocc_A] = C_occ_embedded
            C_full_system[:, nocc_A:nocc] = C_occ_frozen
            C_full_system[:, nocc:] = C_uocc

            Fock, E_low_level_of_theory = self._fock_build_A(
                True, C_full_system, P_frozen, nao, nocc_A, S, H_core, g_ao
            )

            E_new_A = np.einsum("pq,pq->", 2 * H_core, P_of_A, optimize=True)
            E_new_A += np.einsum("pq,pqrs,rs->", 2 * P_of_A, g_ao, P_of_A, optimize=True)
            E_new_A -= np.einsum("pq,prqs,rs->", 1 * P_of_A, g_ao, P_of_A, optimize=True)

            E_new_A += E_low_level_of_theory + E_nuc

            print(
                "SCF Iteration %3d: Energy = %4.16f dE = % 1.5E"
                % (scf_iter, E_new_A, E_new_A - E_old)
            )

            # SCF Converged?
            if abs(E_new_A - E_old) < E_threshold:
                break
            E_old = E_new_A

            if scf_iter == MAXITER:
                raise Exception("Maximum number of SCF iterations exceeded.")

        # Post iterations
        print("\nSCF converged.")
        print("Final SCF A-in-B Energy: %.14f [Eh]" % (E_new_A))

        # post convergence wrapup
        Projector = np.dot(S, np.dot(P_frozen, S))

        Fock, E_low_level_of_theory = self._fock_build_A(
            False, C_full_system, P_frozen, nao, nocc_A, S, H_core, g_ao
        )

        occ_orbital_energy_mat = np.dot(C_occ_embedded.transpose(), np.dot(Fock, C_occ_embedded))
        occ_orbital_energy = np.diag(occ_orbital_energy_mat)

        mu = 1.0e8
        Fock -= mu * Projector

        P_full = P_of_A + P_frozen
        C_occ_projected = np.dot(P_full, np.dot(S, C_uocc))

        if np.linalg.norm(C_occ_projected) < 1e-05:
            print("occupied and unoccupied are orthogonal")
            nonorthogonal = False
        else:
            print("occupied and unoccupied are NOT orthogonal")
            nonorthogonal = True

        # orthogonalization procedure
        if nonorthogonal == True:
            C_uocc_projected = C_uocc - C_occ_projected

            eval, evec = np.linalg.eigh(
                np.dot(C_uocc_projected.transpose(), np.dot(S, C_uocc_projected))
            )

            for i in range(evec.shape[0]):
                eval[i] = eval[i] ** (-0.5)
            eval = np.diag(eval)

            C_uocc = np.dot(C_uocc_projected, np.dot(evec, eval))

            eval_Fock, evec_Fock = np.linalg.eigh(np.dot(C_uocc.transpose(), np.dot(Fock, C_uocc)))
            C_uocc = np.dot(C_uocc, evec_Fock)

        # doing concentric local virtuals
        zeta = 1
        nvir = nmo - nocc
        nvir_act = nvir
        nvir_frozen = 0
        C_uocc_prime, nvir_act, nvir_frozen = self._get_truncated_virtuals(
            S, S, C_uocc, self.num_bf_A, Fock, zeta, S
        )
        # print("nvir_act = %g" %(nvir_act))
        # print("nvir_frozen = %g" % (nvir_frozen))

        C_excld_v = C_uocc_prime[:, nvir_act:]
        proj_excluded_virts = np.dot(S, np.dot(C_excld_v, np.dot(C_excld_v.transpose(), S)))
        Fock += mu * proj_excluded_virts

        C_occ_embedded_cols = C_occ_embedded.shape[1]
        C_full_system_truncated = np.zeros(
            (C_occ_embedded.shape[0], C_occ_embedded.shape[1] + nvir_act)
        )
        C_full_system_truncated[:, :C_occ_embedded_cols] = C_occ_embedded
        C_full_system_truncated[:, C_occ_embedded_cols:] = C_uocc_prime[:, :nvir_act]

        nmo_A_tmp = C_full_system_truncated.shape[1] - nfv
        C_full_system_truncated = C_full_system_truncated[:, nfc:nmo_A_tmp]
        nmo_A = C_full_system_truncated.shape[1]
        print("nmo_A = ", nmo_A)
        nocc_A -= nfc

        orbital_energy_mat = np.dot(
            C_full_system_truncated.transpose(), np.dot(Fock, C_full_system_truncated)
        )
        orbital_energy = np.diag(orbital_energy_mat)
        print(orbital_energy)
        print("")
        print("starting with the WFN-in-SCF calculation")
        nbasis_e_truncated = C_full_system_truncated.shape[1]

        ###Storing integrals to FCIDUMP

        g_mo_sf = PolynomialTensor.einsum(
            {"prsq,pi,qj,rk,sl->iklj": ("++--", *("+-",) * 4, "++--")},
            PolynomialTensor({"++--": g_ao}),
            *(PolynomialTensor({"+-": C_full_system_truncated}, validate=False),) * 4,
        )["++--"]

        # AO2MO of one-body
        h_mo_sf = np.einsum(
            "pi,pq,qj -> ij",
            C_full_system_truncated,
            H_core,
            C_full_system_truncated,
            optimize=True,
        )

        # f = h + 2J-K
        f = np.diag(orbital_energy)
        h_eff = np.zeros((f.shape[0], f.shape[1]))
        h_eff += f
        h_eff -= np.einsum("pqii -> pq", 2.0 * g_mo_sf[:, :, :nocc_A, :nocc_A], optimize=True)
        h_eff += np.einsum("piqi -> pq", g_mo_sf[:, :nocc_A, :, :nocc_A], optimize=True)

        # temporary for validation
        # f_mo_sf = self._get_fock_sf(h_eff, g_mo_sf, nocc_A)
        # nmo_A = C_full_system_truncated.shape[1]

        E_new_A_only = np.einsum("ii->", 2 * h_mo_sf[:nocc_A, :nocc_A], optimize=True)
        E_new_A_only += np.einsum(
            "iijj->", 2 * g_mo_sf[:nocc_A, :nocc_A, :nocc_A, :nocc_A], optimize=True
        )
        E_new_A_only -= np.einsum(
            "ijij->", 1 * g_mo_sf[:nocc_A, :nocc_A, :nocc_A, :nocc_A], optimize=True
        )
        print("Final RHF A Energy        : %.14f [Eh]" % (E_new_A_only))
        E_new_A_only += E_nuc
        print("Final RHF A Energy tot    : %.14f [Eh]" % (E_new_A_only))

        E_new_A_only = np.einsum("ii->", 2 * h_eff[:nocc_A, :nocc_A], optimize=True)
        E_new_A_only += np.einsum(
            "iijj->", 2 * g_mo_sf[:nocc_A, :nocc_A, :nocc_A, :nocc_A], optimize=True
        )
        E_new_A_only -= np.einsum(
            "ijij->", 1 * g_mo_sf[:nocc_A, :nocc_A, :nocc_A, :nocc_A], optimize=True
        )
        print("Final RHF A eff Energy        : %.14f [Eh]" % (E_new_A_only))
        shift = -1.0 * float(E_new_A_only)
        E_new_A_only += E_nuc
        print("Final RHF A eff Energy tot    : %.14f [Eh]" % (E_new_A_only))

        new_hamiltonian = ElectronicEnergy.from_raw_integrals(h_eff, g_mo_sf)
        new_hamiltonian.nuclear_repulsion_energy = float(E_new_A)
        new_hamiltonian.constants["ProjectionTransformer"] = shift

        result = ElectronicStructureProblem(new_hamiltonian)
        result.num_particles = self.num_elec_A - num_fc_elec
        result.num_spatial_orbitals = nmo_A

        g_mo_sf = g_mo_sf.transpose(0, 2, 1, 3)

        e_ij = orbital_energy[:nocc_A]
        e_ab = orbital_energy[nocc_A:]

        # -1 means that it is an unknown dimension and we want numpy to figure it out.
        e_denom = 1 / (
            e_ij.reshape(-1, 1, 1, 1) - e_ab.reshape(-1, 1, 1) + e_ij.reshape(-1, 1) - e_ab
        )

        g_vvoo_ee = g_mo_sf[nocc_A:, nocc_A:, :nocc_A, :nocc_A]
        g_oovv_ee = g_mo_sf[:nocc_A, :nocc_A, nocc_A:, nocc_A:]
        t2_ee = np.einsum("iajb,abij->abij", e_denom, g_vvoo_ee, optimize=True)

        MP2_ee = np.einsum("ikac,acik->", g_oovv_ee, t2_ee, optimize=True)
        # +g_ee(ijab) * t2_ee(abij) * 0.5
        MP2_ee += np.einsum("ijab,abij->", 0.5 * g_oovv_ee, t2_ee, optimize=True)
        # -g_ee(ijab) * t2_ee(abji) * 0.5
        MP2_ee -= np.einsum("ijab,abji->", 0.5 * g_oovv_ee, t2_ee, optimize=True)
        # +g_ee(klcd) * t2_ee(cdkl) * 0.5
        MP2_ee += np.einsum("klcd,cdkl->", 0.5 * g_oovv_ee, t2_ee, optimize=True)
        # -g_ee(klcd) * t2_ee(cdlk) * 0.5
        MP2_ee -= np.einsum("klcd,cdlk->", 0.5 * g_oovv_ee, t2_ee, optimize=True)

        print("MP2_ee = %4.10f" % (MP2_ee))

        return result

    def _spade_partition(self, S, C_occ, Nfrags, nocc, nocc_A, nao_A):
        print("")
        print("Doing SPADE partitioning")
        print("D. CLaudino and N. Mayhall JCTC 15, 1053 (2019)")
        print("")

        S_05_inv = self._symmetric_orthogonalization(S)

        C_tmp = np.dot(S_05_inv, C_occ)
        C_tmp2 = C_tmp[:nao_A, :]

        U, sigma, Vt = np.linalg.svd(C_tmp2, full_matrices=True)
        V = Vt.transpose()

        return np.dot(C_occ, V[:, :nocc_A]), np.dot(C_occ, V[:, nocc_A:])

    def _symmetric_orthogonalization(self, S):
        D, V = np.linalg.eigh(S)
        for i in range(V.shape[0]):
            D[i] = D[i] ** 0.5

        D = np.diag(D)
        X = np.dot(V, np.dot(D, V.transpose()))
        return X

    def _fock_build_A(self, project, C_full_system, P_frozen, nao, nocc_A, S, H_core, I):
        # construction of the Fock matrix
        # Fragment A contribution
        C_of_A = np.zeros((nao, nao))
        C_of_A[:, :nocc_A] = C_full_system[:, :nocc_A]

        P_of_A = C_of_A.dot(C_of_A.transpose())
        J_of_A = np.einsum("pqrs,rs->pq", I, P_of_A, optimize=True)
        K_of_A = np.einsum("prqs,rs->pq", I, P_of_A, optimize=True)
        Fock = H_core + 2 * J_of_A - K_of_A

        P_full = P_of_A + P_frozen
        J_full = np.einsum("pqrs,rs->pq", I, P_full, optimize=True)
        K_full = np.einsum("prqs,rs->pq", I, P_full, optimize=True)
        Fock_tot_low_level = H_core + 2 * J_full - K_full

        Fock_of_A_low_level = H_core + 2 * J_of_A - K_of_A

        Fock += 2 * J_full - K_full - (2 * J_of_A - K_of_A)

        E_low_level_of_theory = np.einsum(
            "pq,pq->", (H_core + Fock_tot_low_level), P_full, optimize=True
        )
        E_low_level_of_theory -= np.einsum(
            "pq,pq->", (H_core + Fock_of_A_low_level), P_of_A, optimize=True
        )

        if project == True:
            Projector = np.identity(nao) - S.dot(P_frozen)
            Fock = np.dot(Projector, np.dot(Fock, Projector.transpose()))

        return Fock, E_low_level_of_theory

    def _get_truncated_virtuals(
        self, working_basis, projection_basis, C_uocc, nao_A, Fock, zeta, S
    ):

        print("")
        print("doing truncated local virtuals ")
        print("    Concentric localization and truncation for virtuals    ")
        print("     D. Claudino and N. Mayhall, JCTC, 15, 6085 (2019)     ")
        print("")

        nao = S.shape[0]
        projection_basis_embed = projection_basis[:nao_A, :nao_A]

        working_basis_red = working_basis[:nao_A, :]

        projection_basis_embed_inv = np.linalg.inv(projection_basis_embed)
        C_uocc_prime = np.dot(projection_basis_embed_inv, np.dot(working_basis_red, C_uocc))
        C_uocc_p_projection_basisWB_C_uocc = np.dot(
            C_uocc_prime.transpose(), np.dot(working_basis_red, C_uocc)
        )

        U, sigma, Vt = np.linalg.svd(C_uocc_p_projection_basisWB_C_uocc, full_matrices=True)
        V = Vt.transpose()

        C_uocc_new = np.dot(C_uocc, V[:, :nao_A])
        C_uocc_rem = np.dot(C_uocc, V[:, nao_A:])

        C_uocc_new = C_uocc_new

        for iter in range(zeta - 1):
            F_new_rem = np.dot(C_uocc_new.transpose(), np.dot(Fock, C_uocc_rem))
            U, sigma, Vt = np.linalg.svd(F_new_rem, full_matrices=True)
            V = Vt.transpose()

            # update
            C_uocc_rem_ncols = C_uocc_rem.shape[1]
            C_uocc_cur_ncols = C_uocc_cur.shape[1]
            if C_uocc_rem_ncols > C_uocc_cur_ncols:
                C_uocc_cur = np.dot(C_uocc_rem, V[:, :C_uocc_cur_ncols])
                C_uocc_rem = np.dot(C_uocc_rem, V[:, C_uocc_cur_ncols:])

            else:
                C_uocc_cur = np.dot(C_uocc_rem, V)
                C_uocc_rem = np.zeros((C_uocc_rem.shape[0], C_uocc_rem.shape[1]))

            C_uocc_new_nrows = C_uocc_new.shape[0]
            C_uocc_new_ncols = C_uocc_new.shape[1]
            C_uocc_cur_ncols = C_uocc_cur.shape[1]
            C_uocc_new_TMP = np.zeros((C_uocc_new_nrows, C_uocc_new_ncols + C_uocc_cur_ncols))
            C_uocc_new_TMP[:, :C_uocc_new_ncols] = C_uocc_new
            C_uocc_new_TMP[:, C_uocc_new_ncols:] = C_uocc_cur

            C_uocc_new = C_uocc_new_TMP

        print("Pseudocanonicalizing the selected and excluded virtuals separately")

        F_new_virtvirt = np.dot(C_uocc_new.transpose(), np.dot(Fock, C_uocc_new))

        e_F_new_virtvirt, U_new = np.linalg.eigh(F_new_virtvirt)

        C_uocc_new = np.dot(C_uocc_new, U_new)

        C_uocc_rem_ncols = C_uocc_rem.shape[1]
        if C_uocc_rem_ncols != 0:
            F_rem_virtvirt = np.dot(C_uocc_rem.transpose(), np.dot(Fock, C_uocc_rem))
            eigs_F_rem_virtvirt, U_rem = np.linalg.eigh(F_rem_virtvirt)

            C_uocc_rem = np.dot(C_uocc_rem, U_rem)
            C_uocc_new_nrows = C_uocc_new.shape[0]
            C_uocc_new_ncols = C_uocc_new.shape[1]
            C_uocc_rem_ncols = C_uocc_rem.shape[1]
            C_uocc_TMP = np.zeros((C_uocc_new_nrows, C_uocc_new_ncols + C_uocc_rem_ncols))
            C_uocc_TMP[:, :C_uocc_new_ncols] = C_uocc_new
            C_uocc_TMP[:, C_uocc_new_ncols:] = C_uocc_rem
            C_uocc = C_uocc_TMP
        else:
            C_uocc = C_uocc_new

        nvir_act = C_uocc_new.shape[1]
        nvir_frozen = C_uocc.shape[1] - nvir_act

        return C_uocc, nvir_act, nvir_frozen

    def _get_fock_sf(self, h, g, nocc):
        fock = np.zeros((h.shape[0], h.shape[1]))
        fock += h
        fock += np.einsum("pqii -> pq", 2.0 * g[:, :, :nocc, :nocc], optimize=True)
        fock -= np.einsum("piqi -> pq", g[:, :nocc, :, :nocc], optimize=True)

        return fock
