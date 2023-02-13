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
from typing import cast

import numpy as np
import scipy.linalg as la

from qiskit_nature.second_q.hamiltonians import ElectronicEnergy, Hamiltonian
from qiskit_nature.second_q.operators import ElectronicIntegrals, PolynomialTensor
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
        num_electrons: int | tuple[int, int],  # the number of electrons in the "active" subsystem A
        num_basis_functions: int,  # the number of basis functions in the "active" subsystem A
        num_frozen_occupied_orbitals: int,  # the number of occupied orbitals to freeze
        num_frozen_virtual_orbitals: int,  # the number of virtual orbitals to freeze
        basis_transformer: BasisTransformer,
    ) -> None:
        """TODO."""
        self.num_electrons = num_electrons
        self.num_basis_functions = num_basis_functions
        self.num_frozen_occupied_orbitals = num_frozen_occupied_orbitals
        self.num_frozen_virtual_orbitals = num_frozen_virtual_orbitals
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
        # TODO: hamiltonian.fock does not work as expected for unrestricted spin systems because the
        # hamiltonian is in the AO basis and, thus, has no beta or beta_alpha 2-body terms
        # associated with it yet...
        if not self.basis_transformer.coefficients.beta.is_empty():
            hamiltonian.electronic_integrals.beta = hamiltonian.electronic_integrals.alpha
            hamiltonian.electronic_integrals.beta_alpha = (
                hamiltonian.electronic_integrals.two_body.alpha
            )

        if isinstance(self.num_electrons, tuple):
            nalpha, nbeta = self.num_electrons
        else:
            nbeta = self.num_electrons // 2
            nalpha = self.num_electrons - nbeta

        num_elec_env = sum(problem.num_particles) - (nalpha + nbeta)
        nao = self.basis_transformer.coefficients.alpha.register_length
        nocc_a = (nalpha + nbeta) // 2
        nocc_b = num_elec_env // 2
        nocc = nocc_a + nocc_b

        mo_coeff_occ_ints, mo_coeff_vir_ints = self.basis_transformer.coefficients.split(
            np.hsplit, [nocc], validate=False
        )

        overlap = problem.overlap_matrix
        overlap[np.abs(overlap) < 1e-12] = 0.0

        # TODO: make localization method configurable
        fragment_a, fragment_b = _spade_partition(
            overlap, mo_coeff_occ_ints, self.num_basis_functions, nocc_a
        )

        # NOTE: now we wrap the overlap matrix into ElectronicIntegrals to simplify handling later
        overlap = ElectronicIntegrals.from_raw_integrals(overlap)

        # NOTE: fragment_a will ONLY change if the SCF loop below is necessary to ensure consistent
        # embedding (which I believe to be trivial in the HF case and, thus, only occur with DFT)

        density_a = ElectronicDensity.einsum({"ij,kj->ik": ("+-",) * 3}, fragment_a, fragment_a)
        density_b = ElectronicDensity.einsum({"ij,kj->ik": ("+-",) * 3}, fragment_b, fragment_b)

        fock_, e_low_level = _fock_build_a(density_a, density_b, hamiltonian)

        identity = ElectronicIntegrals.from_raw_integrals(np.identity(nao))
        projector = identity - ElectronicIntegrals.einsum(
            {"ij,jk->ik": ("+-",) * 3}, overlap, density_b
        )
        fock = ElectronicIntegrals.einsum(
            {"ij,jk,lk->il": ("+-",) * 4}, projector, fock_, projector
        )

        e_old = 0
        # TODO: make these configurable
        e_thres = 1e-7
        max_iter = 50

        logger.info("")
        logger.info(" Hartree-Fock for subsystem A Energy")

        e_nuc = hamiltonian.nuclear_repulsion_energy

        # TODO: is this SCF loop necessary in the HF case?
        # NOTE: yes this is not necessary for HF and only ensures that a *-in-DFT embedding is
        # self-consistent before starting
        # NOTE: we do need at least one iteration here because this computes e_new_a
        # TODO: actually make this SCF loop a standalone method
        for scf_iter in range(1, max_iter + 1):

            # TODO: improve the following 7 lines
            mo_coeff_a_full_alpha: np.ndarray = None
            if "+-" in fock.alpha:
                _, mo_coeff_a_full_alpha = la.eigh(fock.alpha["+-"], overlap.alpha["+-"])

            mo_coeff_a_full_beta: np.ndarray = None
            if "+-" in fock.beta:
                _, mo_coeff_a_full_beta = la.eigh(fock.beta["+-"], overlap.alpha["+-"])

            mo_coeff_a_full = ElectronicIntegrals.from_raw_integrals(
                mo_coeff_a_full_alpha, h1_b=mo_coeff_a_full_beta, validate=False
            )

            fragment_a, _ = mo_coeff_a_full.split(np.hsplit, [nocc_a], validate=False)

            density_a = ElectronicDensity.einsum({"ij,kj->ik": ("+-",) * 3}, fragment_a, fragment_a)

            fock_, e_low_level = _fock_build_a(density_a, density_b, hamiltonian)

            projector = identity - ElectronicIntegrals.einsum(
                {"ij,jk->ik": ("+-",) * 3}, overlap, density_b
            )
            fock = ElectronicIntegrals.einsum(
                {"ij,jk,lk->il": ("+-",) * 4}, projector, fock_, projector
            )

            e_new_a_ints = ElectronicIntegrals.einsum(
                {"ij,ji": ("+-", "+-", "")},
                hamiltonian.electronic_integrals.one_body + hamiltonian.fock(density_a),
                density_a,
            )

            e_new_a = (
                e_new_a_ints.alpha.get("", 0.0)
                + e_new_a_ints.beta.get("", 0.0)
                + e_new_a_ints.beta_alpha.get("", 0.0)
                + e_low_level
                + e_nuc
            )

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

        mu = 1.0e8
        # TODO: what exactly is this step in which we subtract a projector?
        # am I right assuming that this deals with the occupied orbitals of subsystem B?
        # NOTE: indeed this projects the occupied space of the environment (B) into a high energetic
        # space, thereby separating the fragment from its environment
        fock -= mu * ElectronicIntegrals.einsum(
            {"ij,jk,kl->il": ("+-",) * 4}, overlap, density_b, overlap
        )

        density_full = density_a + density_b
        mo_coeff_projected = ElectronicIntegrals.einsum(
            {"ij,jk,kl->il": ("+-",) * 4},
            density_full,
            overlap,
            mo_coeff_vir_ints,
            validate=False,
        )

        # TODO: this check is not yet done for UHF because it only applies to DFT case anyways
        if np.linalg.norm(mo_coeff_projected.alpha["+-"]) < 1e-05:
            logger.info("occupied and unoccupied are orthogonal")
            nonorthogonal = False
        else:
            logger.info("occupied and unoccupied are NOT orthogonal")
            nonorthogonal = True

        # orthogonalization procedure
        # TODO: which procedure is this exactly and when does it become necessary? In the DFT case?
        # NOTE: indeed, this is not triggered in the HF-case because the orbitals are already
        # orthogonal (at least for systems known to us)
        # TODO: extract into a separate method because this is only necessary for DFT
        if nonorthogonal:
            mo_coeff_vir_projected = mo_coeff_vir_ints - mo_coeff_projected

            einsummed = ElectronicIntegrals.einsum(
                {"ji,jk,kl->il": ("+-",) * 4},
                mo_coeff_vir_projected,
                overlap,
                mo_coeff_vir_projected,
                validate=False,
            )

            # TODO: improve the following 7 lines
            eigval_alpha, eigvec_alpha = None, None
            if "+-" in einsummed.alpha:
                eigval_alpha, eigvec_alpha = np.linalg.eigh(einsummed.alpha["+-"])
                eigval_alpha = np.linalg.inv(np.diag(np.sqrt(eigval_alpha)))

            eigval_beta, eigvec_beta = None, None
            if "+-" in einsummed.beta:
                eigval_beta, eigvec_beta = np.linalg.eigh(einsummed.beta["+-"])
                eigval_beta = np.linalg.inv(np.diag(np.sqrt(eigval_beta)))

            eigvec = ElectronicIntegrals.from_raw_integrals(
                eigvec_alpha, h1_b=eigvec_beta, validate=False
            )
            eigval = ElectronicIntegrals.from_raw_integrals(
                eigval_alpha, h1_b=eigval_beta, validate=False
            )

            mo_coeff_vir_ints = ElectronicIntegrals.einsum(
                {"ij,jk,kl->il": ("+-",) * 4},
                mo_coeff_vir_projected,
                eigvec,
                eigval,
                validate=False,
            )

            einsummed = ElectronicIntegrals.einsum(
                {"ji,jk,kl->il": ("+-",) * 4},
                mo_coeff_vir_ints,
                fock,
                mo_coeff_vir_ints,
                validate=False,
            )

            # TODO: improve the following 7 lines
            eigvec_fock_alpha = None
            if "+-" in einsummed.alpha:
                _, eigvec_fock_alpha = np.linalg.eigh(einsummed.alpha["+-"])

            eigvec_fock_beta = None
            if "+-" in einsummed.beta:
                _, eigvec_fock_beta = np.linalg.eigh(einsummed.beta["+-"])

            eigvec_fock = ElectronicIntegrals.from_raw_integrals(
                eigvec_fock_alpha, h1_b=eigvec_fock_beta, validate=False
            )

            mo_coeff_vir_ints = ElectronicIntegrals.einsum(
                {"ij,jk->ik": ("+-",) * 3}, mo_coeff_vir_ints, eigvec_fock, validate=False
            )

        # doing concentric local virtuals
        (
            mo_coeff_vir_pb,
            (nvir_a_alpha, nvir_a_beta),
            (nvir_b_alpha, nvir_b_beta),
        ) = _concentric_localization(
            overlap.split(np.vsplit, [self.num_basis_functions], validate=False)[0],
            overlap.split(np.vsplit, [self.num_basis_functions], validate=False)[0].split(
                np.hsplit, [self.num_basis_functions], validate=False
            )[0],
            mo_coeff_vir_ints,
            self.num_basis_functions,
            fock,
            zeta=1,  # TODO: make configurable and figure out what exactly zeta is meant to do?
        )
        logger.debug("nvir_a_alpha = %s", nvir_a_alpha)
        logger.debug("nvir_b_alpha = %s", nvir_b_alpha)
        logger.debug("nvir_a_beta = %s", nvir_a_beta)
        logger.debug("nvir_b_beta = %s", nvir_b_beta)

        mo_coeff_vir_a_alpha, mo_coeff_vir_b_alpha = mo_coeff_vir_pb.alpha.split(
            np.hsplit, [nvir_a_alpha], validate=False
        )

        mo_coeff_vir_a_beta, mo_coeff_vir_b_beta = None, None
        if "+-" in mo_coeff_vir_pb.beta:
            mo_coeff_vir_a_beta, mo_coeff_vir_b_beta = mo_coeff_vir_pb.beta.split(
                np.hsplit, [nvir_a_beta], validate=False
            )

        mo_coeff_vir_a = ElectronicIntegrals(mo_coeff_vir_a_alpha, mo_coeff_vir_a_beta)
        mo_coeff_vir_b = ElectronicIntegrals(mo_coeff_vir_b_alpha, mo_coeff_vir_b_beta)

        # P^B in Manby2012
        proj_excluded_virts = ElectronicIntegrals.einsum(
            {"ij,jk,lk,lm->im": ("+-",) * 5},
            overlap,
            mo_coeff_vir_b,
            mo_coeff_vir_b,
            overlap,
            validate=False,
        )
        fock += mu * proj_excluded_virts

        max_orb = -self.num_frozen_virtual_orbitals if self.num_frozen_virtual_orbitals else None

        _, mo_coeff_final, _ = ElectronicIntegrals.stack(
            np.hstack,
            (fragment_a, mo_coeff_vir_a),
            validate=False,
        ).split(np.hsplit, [self.num_frozen_occupied_orbitals, max_orb], validate=False)

        nmo_a = mo_coeff_final.alpha["+-"].shape[1]
        logger.debug("nmo_a = %s", nmo_a)

        if "+-" in mo_coeff_final.beta and nmo_a != mo_coeff_final.beta["+-"].shape[1]:
            raise NotImplementedError("TODO.")

        nocc_a -= self.num_frozen_occupied_orbitals

        # NOTE: at this point fock =  (omitting pre-factors for Coulomb to be RKS/UKS-agnostic)
        #   h_core + J_A - K_A
        #   + J_tot - xc_fac * K_tot + Vxc_tot
        #   - (J_A - xc_fac * K_A + Vxc_A)
        #   - mu * P_B_occ
        #   + mu * P_B_vir
        # NOTE: this matches Eq. (3) from Many2012 with the additional J_A and K_A terms;
        # these are included to take the 2-body terms into account
        orbital_energy = ElectronicIntegrals.apply(
            lambda arr: np.diag(np.diag(arr)),
            ElectronicIntegrals.einsum(
                {"ji,jk,kl->il": ("+-",) * 4}, mo_coeff_final, fock, mo_coeff_final, validate=False
            ),
        )
        logger.info("alpha orbital energies")
        logger.info(orbital_energy.alpha)
        if "+-" in mo_coeff_final.beta:
            logger.info("beta orbital energies")
            logger.info(orbital_energy.beta)

        transform = BasisTransformer(ElectronicBasis.AO, ElectronicBasis.MO, mo_coeff_final)

        new_hamiltonian = cast(ElectronicEnergy, transform.transform_hamiltonian(hamiltonian))
        # now, new_hamiltonian is simply the hamiltonian we started with but transformed into the
        # projected MO basis (which is limited to subsystem A)

        only_a = ElectronicDensity.from_raw_integrals(
            np.diag([1.0 if i < nocc_a else 0.0 for i in range(nmo_a)]),
            # TODO: extend nocc_a to distinguish alpha and beta
            h1_b=np.diag([1.0 if i < nocc_a else 0.0 for i in range(nmo_a)])
            if not mo_coeff_final.beta.is_empty()
            else None,
        )

        e_new_a_only = ElectronicIntegrals.einsum(
            {"ij,ji": ("+-", "+-", "")},
            new_hamiltonian.electronic_integrals.one_body + new_hamiltonian.fock(only_a),
            only_a,
        )

        e_new_a_only = (
            e_new_a_only.alpha.get("", 0.0)
            + e_new_a_only.beta.get("", 0.0)
            + e_new_a_only.beta_alpha.get("", 0.0)
        )

        logger.info("Final RHF A Energy        : %.14f [Eh]", e_new_a_only)
        logger.info("Final RHF A Energy tot    : %.14f [Eh]", e_new_a_only + e_nuc)

        # next, we subtract the fock operator from the hamiltonian which subtracts h + J_A - K_A
        new_hamiltonian.electronic_integrals -= new_hamiltonian.fock(only_a)
        # and finally we add the diagonal matrix containing the orbital energies
        new_hamiltonian.electronic_integrals += orbital_energy
        # TODO: to verify: this re-adds the contributions from h + J_tot - K_tot + \mu * P_B
        # this is a trick to avoid double counting
        # new_hamiltonian now equals h_{A in B} (see Eq. (3) in Manby2012)

        e_new_a_only = ElectronicIntegrals.einsum(
            {"ij,ji": ("+-", "+-", "")},
            new_hamiltonian.electronic_integrals.one_body + new_hamiltonian.fock(only_a),
            only_a,
        )

        e_new_a_only = (
            e_new_a_only.alpha.get("", 0.0)
            + e_new_a_only.beta.get("", 0.0)
            + e_new_a_only.beta_alpha.get("", 0.0)
        )

        logger.info("Final RHF A eff Energy        : %.14f [Eh]", e_new_a_only)
        logger.info("Final RHF A eff Energy tot    : %.14f [Eh]", e_new_a_only + e_nuc)

        print(e_new_a, e_new_a_only)
        new_hamiltonian.nuclear_repulsion_energy = float(e_new_a)
        new_hamiltonian.constants["ProjectionTransformer"] = -1.0 * float(e_new_a_only)

        if logger.isEnabledFor(logging.INFO):

            from qiskit_nature.second_q.algorithms.initial_points.mp2_initial_point import (
                _compute_mp2,
            )

            # TODO: extend to unrestricted spin once MP2 supports that
            _, e_mp2 = _compute_mp2(
                nocc_a,
                new_hamiltonian.electronic_integrals.two_body.alpha["++--"],
                np.diag(orbital_energy.alpha["+-"]),
            )

            logger.info("e_mp2 = %4.10f", e_mp2)

        result = ElectronicStructureProblem(new_hamiltonian)
        result.num_particles = (nalpha + nbeta) - (self.num_frozen_occupied_orbitals * 2)
        result.num_spatial_orbitals = nmo_a

        return result


def _fock_build_a(
    density_a: ElectronicDensity, density_b: ElectronicDensity, hamiltonian: ElectronicEnergy
):
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

    return fock_final, (
        e_low_level.alpha.get("", 0.0)
        + e_low_level.beta.get("", 0.0)
        + e_low_level.beta_alpha.get("", 0.0)
    )


def _concentric_localization(overlap_pb_wb, projection_basis, mo_coeff_vir, num_bf, fock, zeta):
    logger.info("")
    logger.info("Doing concentric location and truncation of virtual space")
    logger.info("    Concentric localization and truncation for virtuals    ")
    logger.info("     D. Claudino and N. Mayhall, JCTC, 15, 6085 (2019)     ")
    logger.info("")

    # S^{-1} in paper
    overlap_a_pb_inv = ElectronicIntegrals.apply(np.linalg.inv, projection_basis, validate=False)

    # C'_{vir} in paper
    mo_coeff_vir_pb = ElectronicIntegrals.einsum(
        {"ij,jk,kl->il": ("+-",) * 4}, overlap_a_pb_inv, overlap_pb_wb, mo_coeff_vir, validate=False
    )

    # Eq. (10a)
    einsummed = ElectronicIntegrals.einsum(
        {"ji,jk,kl->il": ("+-",) * 4}, mo_coeff_vir_pb, overlap_pb_wb, mo_coeff_vir, validate=False
    )
    # TODO: improve the following 7 lines
    v_t_alpha: np.ndarray = None
    if "+-" in einsummed.alpha:
        _, _, v_t_alpha = np.linalg.svd(einsummed.alpha["+-"], full_matrices=True)

    v_t_beta: np.ndarray = None
    if "+-" in einsummed.beta:
        _, _, v_t_beta = np.linalg.svd(einsummed.beta["+-"], full_matrices=True)

    v_t = ElectronicIntegrals.from_raw_integrals(v_t_alpha, h1_b=v_t_beta, validate=False)

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

    # zeta controls the number of unoccupied orbitals
    # for _ in range(zeta - 1):
    #     einsummed = ElectronicIntegrals.einsum(
    #         {"ji,jk,kl->il": ("+-",) * 4}, mo_coeff_vir_new, fock, mo_coeff_vir_kern, validate=False
    #     )
    #     # Eq. (12a)
    #     _, _, v_t = np.linalg.svd(
    #         einsummed.alpha["+-"],
    #         full_matrices=True,
    #     )
    #
    #     # Eq. (12b)
    #     # TODO: fix bug w.r.t. undefined mo_coeff_vir_cur
    #     v_span, v_kern = np.hsplit(v_t.T, [mo_coeff_vir_cur.shape[1]])
    #
    #     # Eq. (12c-12d)
    #     if mo_coeff_vir_kern.shape[1] > mo_coeff_vir_cur.shape[1]:
    #         mo_coeff_vir_cur = np.dot(mo_coeff_vir_kern, v_span)
    #         mo_coeff_vir_kern = np.dot(mo_coeff_vir_kern, v_kern)
    #
    #     else:
    #         mo_coeff_vir_cur = np.dot(mo_coeff_vir_kern, v_t.T)
    #         mo_coeff_vir_kern = np.zeros_like(mo_coeff_vir_kern)
    #
    #     # Eq. (12e)
    #     mo_coeff_vir_new = np.hstack((mo_coeff_vir_new, mo_coeff_vir_cur))

    # post-processing step
    logger.info("Pseudocanonicalizing the selected and excluded virtuals separately")

    einsummed = ElectronicIntegrals.einsum(
        {"ji,jk,kl->il": ("+-",) * 4}, mo_coeff_vir_new, fock, mo_coeff_vir_new, validate=False
    )

    # TODO: improve the following 7 lines
    _, eigvec_alpha = None, None
    if "+-" in einsummed.alpha:
        _, eigvec_alpha = np.linalg.eigh(einsummed.alpha["+-"])

    _, eigvec_beta = None, None
    if "+-" in einsummed.beta:
        _, eigvec_beta = np.linalg.eigh(einsummed.beta["+-"])

    eigvec = ElectronicIntegrals.from_raw_integrals(eigvec_alpha, h1_b=eigvec_beta, validate=False)

    mo_coeff_vir_new = ElectronicIntegrals.einsum(
        {"ij,jk->ik": ("+-",) * 3}, mo_coeff_vir_new, eigvec, validate=False
    )

    mo_coeff_vir_kern_alpha = mo_coeff_vir_kern.alpha
    if "+-" in mo_coeff_vir_kern_alpha and mo_coeff_vir_kern_alpha["+-"].shape[1] != 0:
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
            np.hstack, (mo_coeff_vir_new.alpha, mo_coeff_vir_kern_alpha), validate=False
        )
    else:
        mo_coeff_vir_alpha = mo_coeff_vir_new.alpha

    mo_coeff_vir_kern_beta = mo_coeff_vir_kern.beta
    if "+-" in mo_coeff_vir_kern_beta and mo_coeff_vir_kern_beta["+-"].shape[1] != 0:
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
            np.hstack, (mo_coeff_vir_new.beta, mo_coeff_vir_kern_beta), validate=False
        )
    else:
        mo_coeff_vir_beta = mo_coeff_vir_new.beta

    mo_coeff_vir = ElectronicIntegrals(mo_coeff_vir_alpha, mo_coeff_vir_beta)

    nvir_a_alpha = mo_coeff_vir_new.alpha["+-"].shape[1]
    nvir_b_alpha = mo_coeff_vir.alpha["+-"].shape[1] - nvir_a_alpha

    # TODO: we probably need to ensure that the number of virtual orbitals are identical in the
    # alpha and beta cases because none of Qiskit Nature is tested for differing numbers of orbitals
    # per spin!
    nvir_a_beta, nvir_b_beta = None, None
    if "+-" in mo_coeff_vir_new.beta:
        nvir_a_beta = mo_coeff_vir_new.beta["+-"].shape[1]
        nvir_b_beta = mo_coeff_vir.beta["+-"].shape[1] - nvir_a_beta

    return mo_coeff_vir, (nvir_a_alpha, nvir_a_beta), (nvir_b_alpha, nvir_b_beta)


def _spade_partition(
    overlap: np.ndarray, mo_coeff_occ: ElectronicIntegrals, num_bf: int, nocc_a: int
):
    logger.info("")
    logger.info("Doing SPADE partitioning")
    logger.info("D. CLaudino and N. Mayhall JCTC 15, 1053 (2019)")
    logger.info("")

    # 1. use symmetric orthogonalization on the overlap matrix
    symm_orth = ElectronicIntegrals.from_raw_integrals(symmetric_orthogonalization(overlap))

    # 2. change the MO basis to be orthogonal and reasonably localized
    mo_coeff_tmp = ElectronicIntegrals.einsum(
        {"ij,jk->ik": ("+-",) * 3}, symm_orth, mo_coeff_occ, validate=False
    )

    # 3. select the active sector
    mo_coeff_tmp, _ = mo_coeff_tmp.split(np.vsplit, [num_bf], validate=False)

    # 4. use SVD to find the final rotation matrix
    # TODO: improve the following 7 lines
    rot_alpha: np.ndarray = None
    if "+-" in mo_coeff_tmp.alpha:
        _, _, rot_alpha = np.linalg.svd(mo_coeff_tmp.alpha["+-"], full_matrices=True)

    rot_beta: np.ndarray = None
    if "+-" in mo_coeff_tmp.beta:
        _, _, rot_beta = np.linalg.svd(mo_coeff_tmp.beta["+-"], full_matrices=True)

    rot = ElectronicIntegrals.from_raw_integrals(rot_alpha, h1_b=rot_beta, validate=False)
    rot_t = ElectronicIntegrals.apply(np.transpose, rot, validate=False)
    left, right = rot_t.split(np.hsplit, [nocc_a], validate=False)

    return (
        ElectronicIntegrals.einsum({"ij,jk->ik": ("+-",) * 3}, mo_coeff_occ, left, validate=False),
        ElectronicIntegrals.einsum({"ij,jk->ik": ("+-",) * 3}, mo_coeff_occ, right, validate=False),
    )
