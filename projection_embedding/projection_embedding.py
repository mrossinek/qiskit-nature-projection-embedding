# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Projection Embedding."""

from __future__ import annotations

import logging
from functools import partial
from typing import Callable, cast

import numpy as np
import scipy.linalg as la

from qiskit_nature.second_q.hamiltonians import ElectronicEnergy, Hamiltonian
from qiskit_nature.second_q.operators import ElectronicIntegrals, PolynomialTensor
from qiskit_nature.second_q.problems import (
    BaseProblem,
    ElectronicBasis,
    ElectronicStructureProblem,
)
from qiskit_nature.second_q.properties import (
    AngularMomentum,
    ElectronicDensity,
    Magnetization,
    ParticleNumber,
)

from qiskit_nature.second_q.transformers import BaseTransformer, BasisTransformer

from .occupied_orbital_partitioning import (
    OccupiedOrbitalPartitioning,
    SPADEPartitioning,
)
from .utils import split_elec_ints_per_spin

logger = logging.getLogger(__name__)


class ProjectionEmbedding(BaseTransformer):
    """TODO."""

    def __init__(
        self,
        # the number of electrons in the "active" subsystem A
        num_electrons: int | tuple[int, int],
        # the number of basis functions in the "active" subsystem A
        num_basis_functions: int,
        basis_transformer: BasisTransformer,
        overlap_matrix: np.ndarray,
        num_active_electrons: int | tuple[int, int] | None = None,
        num_active_orbitals: int | None = None,
        *,
        occupied_orbital_partitioning: OccupiedOrbitalPartitioning = SPADEPartitioning(),
    ) -> None:
        """TODO."""
        self.num_electrons = num_electrons
        self.num_basis_functions = num_basis_functions

        self.num_active_orbitals = num_active_orbitals
        self.num_frozen_occupied_orbitals: int | None = None

        if num_active_electrons is not None:
            if type(num_active_electrons) != type(num_electrons):
                raise TypeError(
                    "The types of the 'num_electrons' and 'num_active_electrons' must match. "
                    f"However, you provided types {type(num_electrons)} and "
                    f"{type(num_active_electrons)} which are incompatible."
                )

            if isinstance(num_active_electrons, int):
                self.num_frozen_occupied_orbitals = (
                    num_electrons - num_active_electrons
                ) // 2

            elif isinstance(num_active_electrons, tuple):
                num_frozen_alpha = num_electrons[0] - num_active_electrons[0]
                num_frozen_beta = num_electrons[1] - num_active_electrons[1]

                if num_frozen_alpha != num_frozen_beta:
                    raise ValueError(
                        "The number of frozen alpha- and beta-spin orbitals must turn out to be "
                        "identical. Their values are computed from the difference of the total "
                        "number of electrons in the embedded fragment and the active number of "
                        "electrons. For your current configuration they turned out to be: "
                        f"{num_frozen_alpha} and {num_frozen_beta}. Please correct your input!"
                    )

                self.num_frozen_occupied_orbitals = num_frozen_alpha

        self.basis_transformer = basis_transformer
        self.overlap = ElectronicIntegrals.from_raw_integrals(overlap_matrix)

        self.occupied_orbital_partitioning = occupied_orbital_partitioning

    def transform(self, problem: BaseProblem) -> BaseProblem:
        """TODO."""
        if isinstance(problem, ElectronicStructureProblem):
            if problem.basis != ElectronicBasis.AO:
                raise ValueError(
                    f"The problem description must be in the AO basis, not {problem.basis.value}."
                )

            return self._transform_electronic_structure_problem(problem)
        else:
            raise TypeError(
                f"The problem of type, {type(problem)}, is not supported by this embedding."
            )

    def transform_hamiltonian(self, hamiltonian: Hamiltonian) -> Hamiltonian:
        """TODO."""
        if isinstance(hamiltonian, ElectronicEnergy):
            pass
        else:
            raise TypeError(
                f"The hamiltonian of type, {type(hamiltonian)}, is not supported by this "
                "embedding."
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

        self.hamiltonian = problem.hamiltonian
        # TODO: hamiltonian.fock does not work as expected for unrestricted spin systems because the
        # hamiltonian is in the AO basis and, thus, has no beta or beta_alpha 2-body terms
        # associated with it yet...
        # NOTE: this will be fixed by https://github.com/qiskit-community/qiskit-nature/pull/1216
        if not self.basis_transformer.coefficients.beta.is_empty():
            self.hamiltonian.electronic_integrals.beta = (
                self.hamiltonian.electronic_integrals.alpha
            )
            self.hamiltonian.electronic_integrals.beta_alpha = (
                self.hamiltonian.electronic_integrals.two_body.alpha
            )

        if isinstance(self.num_electrons, tuple):
            nocc_a_alpha, nocc_a_beta = self.num_electrons
        else:
            nocc_a_beta = self.num_electrons // 2
            nocc_a_alpha = self.num_electrons - nocc_a_beta

        nao = self.basis_transformer.coefficients.register_length
        nocc_b_alpha = problem.num_alpha - nocc_a_alpha
        nocc_b_beta = problem.num_beta - nocc_a_beta

        mo_coeff_occ_ints, mo_coeff_vir_ints = split_elec_ints_per_spin(
            self.basis_transformer.coefficients,
            np.hsplit,
            [nocc_a_alpha + nocc_b_alpha],
            [nocc_a_beta + nocc_b_beta],
        )

        fragment_a, fragment_b = self.occupied_orbital_partitioning.partition(
            self.overlap,
            mo_coeff_occ_ints,
            self.num_basis_functions,
            (nocc_a_alpha, nocc_a_beta),
        )

        # NOTE: fragment_a will ONLY change if the SCF loop below is necessary to ensure consistent
        # embedding (which I believe to be trivial in the HF case and, thus, only occur with DFT)

        density_a = ElectronicDensity.einsum(
            {"ij,kj->ik": ("+-",) * 3}, fragment_a, fragment_a
        )
        if density_a.beta.is_empty():
            density_a.beta = density_a.alpha
        density_b = ElectronicDensity.einsum(
            {"ij,kj->ik": ("+-",) * 3}, fragment_b, fragment_b
        )
        if density_b.beta.is_empty():
            density_b.beta = density_b.alpha

        fock, e_low_level = self._fock_build_a(density_a, density_b)

        e_new_a_ints = 0.5 * ElectronicIntegrals.einsum(
            {"ij,ji": ("+-", "+-", "")},
            self.hamiltonian.electronic_integrals.one_body
            + self.hamiltonian.fock(density_a),
            density_a,
        )

        e_nuc = self.hamiltonian.nuclear_repulsion_energy

        e_new_a = (
            e_new_a_ints.alpha.get("", 0.0)
            + e_new_a_ints.beta.get("", 0.0)
            + e_new_a_ints.beta_alpha.get("", 0.0)
            + e_low_level
            + e_nuc
        )

        identity = ElectronicIntegrals.from_raw_integrals(
            np.identity(nao),
            h1_b=None if density_b.beta.is_empty() else np.identity(nao),
        )
        projector = identity - ElectronicIntegrals.einsum(
            {"ij,jk->ik": ("+-",) * 3}, self.overlap, density_b
        )
        fock = ElectronicIntegrals.einsum(
            {"ij,jk,lk->il": ("+-",) * 4}, projector, fock, projector
        )

        logger.info("")
        logger.info(" Hartree-Fock for subsystem A Energy")

        fock, density_a, fragment_a, e_new_a = self._run_scf_in_scf(
            fock, nocc_a_alpha, nocc_a_beta, density_a, density_b, identity, projector
        )

        # post convergence wrapup
        fock, e_low_level = self._fock_build_a(density_a, density_b)

        mu = 1.0e8
        fock -= mu * ElectronicIntegrals.einsum(
            {"ij,jk,kl->il": ("+-",) * 4}, self.overlap, density_b, self.overlap
        )

        density_full = density_a + density_b
        mo_coeff_projected = ElectronicIntegrals.einsum(
            {"ij,jk,kl->il": ("+-",) * 4},
            density_full,
            self.overlap,
            mo_coeff_vir_ints,
            validate=False,
        )

        mo_coeff_vir_ints = self._orthogonalize(
            mo_coeff_vir_ints, mo_coeff_projected, fock
        )

        # doing concentric local virtuals
        (
            mo_coeff_vir_pb,
            (nvir_a_alpha, nvir_a_beta),
            (nvir_b_alpha, nvir_b_beta),
        ) = _concentric_localization(
            self.overlap.split(np.vsplit, [self.num_basis_functions], validate=False)[
                0
            ],
            self.overlap.split(np.vsplit, [self.num_basis_functions], validate=False)[
                0
            ].split(np.hsplit, [self.num_basis_functions], validate=False)[0],
            mo_coeff_vir_ints,
            self.num_basis_functions,
            fock,
        )

        # NOTE: we need to correct the number of virtual alpha-spin orbitals in fragment A to take a
        # potential number of unpaired electrons into account
        nocc_a_delta = nocc_a_alpha - nocc_a_beta
        if nvir_a_alpha >= nocc_a_delta:
            nvir_a_alpha -= nocc_a_delta
        else:
            nvir_a_beta += nocc_a_delta

        if "+-" in mo_coeff_vir_pb.beta:
            nocc_a_alpha = fragment_a.alpha["+-"].shape[1]
            nocc_a_beta = fragment_a.beta["+-"].shape[1]

        mo_coeff_vir_a, mo_coeff_vir_b = split_elec_ints_per_spin(
            mo_coeff_vir_pb, np.hsplit, [nvir_a_alpha], [nvir_a_beta]
        )

        # P^B in Manby2012
        proj_excluded_virts = ElectronicIntegrals.einsum(
            {"ij,jk,lk,lm->im": ("+-",) * 5},
            self.overlap,
            mo_coeff_vir_b,
            mo_coeff_vir_b,
            self.overlap,
            validate=False,
        )
        fock += mu * proj_excluded_virts

        mo_coeff_final = ElectronicIntegrals.stack(
            np.hstack, (fragment_a, mo_coeff_vir_a), validate=False
        )

        do_split = False
        min_orb, max_orb = None, None
        if self.num_frozen_occupied_orbitals is not None:
            min_orb = self.num_frozen_occupied_orbitals
            do_split = True
        if self.num_active_orbitals is not None:
            if min_orb is None:
                min_orb = 0
            max_orb = min_orb + self.num_active_orbitals
            do_split = True
        if do_split:
            _, mo_coeff_final, _ = mo_coeff_final.split(
                np.hsplit, [min_orb, max_orb], validate=False
            )

        nmo_a = mo_coeff_final.alpha["+-"].shape[1]

        if "+-" in mo_coeff_final.beta:
            nmo_b = mo_coeff_final.beta["+-"].shape[1]
            if nmo_a != nmo_b:
                raise RuntimeError(
                    "The projection embedding resulted in differing numbers of alpha- and beta-spin"
                    f" orbitals: {nmo_a} and {nmo_b}. This scenario is not supported! Please check "
                    "your configuration settings for consistency."
                )

        if self.num_frozen_occupied_orbitals is not None:
            nocc_a_alpha -= self.num_frozen_occupied_orbitals
            nocc_a_beta -= self.num_frozen_occupied_orbitals

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
                {"ji,jk,kl->il": ("+-",) * 4},
                mo_coeff_final,
                fock,
                mo_coeff_final,
                validate=False,
            ),
        )

        self.mo_coeff_final = mo_coeff_final
        transform = BasisTransformer(
            ElectronicBasis.AO, ElectronicBasis.MO, mo_coeff_final
        )

        new_hamiltonian = cast(
            ElectronicEnergy, transform.transform_hamiltonian(self.hamiltonian)
        )
        # now, new_hamiltonian is simply the hamiltonian we started with but transformed into the
        # projected MO basis (which is limited to subsystem A)

        only_a = ElectronicDensity.from_raw_integrals(
            np.diag([1.0 if i < nocc_a_alpha else 0.0 for i in range(nmo_a)]),
            h1_b=np.diag([1.0 if i < nocc_a_beta else 0.0 for i in range(nmo_a)])
            if not mo_coeff_final.beta.is_empty()
            else None,
        )

        e_new_a_only = 0.5 * ElectronicIntegrals.einsum(
            {"ij,ji": ("+-", "+-", "")},
            new_hamiltonian.electronic_integrals.one_body
            + new_hamiltonian.fock(only_a),
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
        # NOTE: this re-adds the contributions from h + J_tot - K_tot + \mu * P_B
        # this is a trick to avoid double counting
        # new_hamiltonian now equals h_{A in B} (see Eq. (3) in Manby2012)

        e_new_a_only = 0.5 * ElectronicIntegrals.einsum(
            {"ij,ji": ("+-", "+-", "")},
            new_hamiltonian.electronic_integrals.one_body
            + new_hamiltonian.fock(only_a),
            only_a,
        )

        e_new_a_only = (
            e_new_a_only.alpha.get("", 0.0)
            + e_new_a_only.beta.get("", 0.0)
            + e_new_a_only.beta_alpha.get("", 0.0)
        )

        logger.info("Final RHF A eff Energy        : %.14f [Eh]", e_new_a_only)
        logger.info("Final RHF A eff Energy tot    : %.14f [Eh]", e_new_a_only + e_nuc)

        new_hamiltonian.nuclear_repulsion_energy = float(e_new_a)
        new_hamiltonian.constants["ProjectionEmbedding"] = -1.0 * float(e_new_a_only)

        result = ElectronicStructureProblem(new_hamiltonian)
        result.num_particles = (nocc_a_alpha, nocc_a_beta)
        result.num_spatial_orbitals = nmo_a
        result.orbital_energy = np.diag(orbital_energy.alpha["+-"])
        if not orbital_energy.beta.is_empty():
            result.orbital_energy_b = np.diag(orbital_energy.beta["+-"])

        for prop in problem.properties:
            if isinstance(prop, (AngularMomentum, Magnetization, ParticleNumber)):
                result.properties.add(prop.__class__(result.num_spatial_orbitals))
            else:
                logger.warning(
                    "Encountered an unsupported property of type '%s'.", type(prop)
                )

        return result

    def _run_scf_in_scf(
        self, fock, nocc_a_alpha, nocc_a_beta, density_a, density_b, identity, projector
    ):
        e_old = 0
        # TODO: make these configurable
        e_thres = 1e-7
        max_iter = 50
        fock_list = []
        diis_error = []

        # TODO: actually make this SCF loop a standalone method
        for scf_iter in range(1, max_iter + 1):
            _, mo_coeff_a_full = ElectronicIntegrals.apply(
                la.eigh, fock, self.overlap, multi=True, validate=False
            )

            fragment_a, _ = split_elec_ints_per_spin(
                mo_coeff_a_full, np.hsplit, [nocc_a_alpha], [nocc_a_beta]
            )

            density_a = ElectronicDensity.einsum(
                {"ij,kj->ik": ("+-",) * 3}, fragment_a, fragment_a
            )

            fock, e_low_level = self._fock_build_a(density_a, density_b)

            projector = identity - ElectronicIntegrals.einsum(
                {"ij,jk->ik": ("+-",) * 3}, self.overlap, density_b
            )
            fock = ElectronicIntegrals.einsum(
                {"ij,jk,lk->il": ("+-",) * 4}, projector, fock, projector
            )

            e_new_a_ints = 0.5 * ElectronicIntegrals.einsum(
                {"ij,ji": ("+-", "+-", "")},
                self.hamiltonian.electronic_integrals.one_body
                + self.hamiltonian.fock(density_a),
                density_a,
            )

            e_new_a = (
                e_new_a_ints.alpha.get("", 0.0)
                + e_new_a_ints.beta.get("", 0.0)
                + e_new_a_ints.beta_alpha.get("", 0.0)
                + e_low_level
                + self.hamiltonian.nuclear_repulsion_energy
            )

            diis_e = ElectronicIntegrals.einsum(
                {"ij,jk,kl->il": ("+-",) * 4}, fock, density_a, self.overlap
            ) - ElectronicIntegrals.einsum(
                {"ij,jk,kl->il": ("+-",) * 4}, self.overlap, density_a, fock
            )
            diis_e = ElectronicIntegrals.einsum(
                {"ij,jk,kl->il": ("+-",) * 4}, projector, diis_e, projector
            )

            fock_list.append(fock)
            diis_error.append(diis_e)
            dRMS_a = np.mean(diis_e.one_body.alpha["+-"] ** 2) ** 0.5
            dRMS_b = np.mean(diis_e.one_body.beta["+-"] ** 2) ** 0.5

            logger.info(
                "SCF Iteration %s: Energy = %s dE = %s dRMS = %s %s",
                scf_iter,
                e_new_a,
                e_new_a - e_old,
                dRMS_a,
                dRMS_b,
            )

            # SCF Converged?
            if abs(e_new_a - e_old) < e_thres and (dRMS_a < 1e-3 and dRMS_b < 1e-3):
                break
            e_old = e_new_a

            if scf_iter >= 2:
                # DIIS

                diis_count = len(fock_list)
                if diis_count > 6:
                    del fock_list[0]
                    del diis_error[0]
                    diis_count -= 1

                # TODO: improve the following implementation, possible leveraging ElectronicIntegrals
                B_a = np.empty((diis_count + 1, diis_count + 1))
                B_a[-1, :] = -1
                B_a[:, -1] = -1
                B_a[-1, -1] = 0
                for num1, e1 in enumerate(diis_error):
                    for num2, e2 in enumerate(diis_error):
                        if num2 > num1:
                            continue
                        val = np.einsum("ij,ij->", e1.alpha["+-"], e2.alpha["+-"])
                        B_a[num1, num2] = val
                        B_a[num2, num1] = val

                B_a[:-1, :-1] /= np.abs(B_a[:-1, :-1]).max()

                resid_a = np.zeros(diis_count + 1)
                resid_a[-1] = -1

                ci_a = np.linalg.solve(B_a, resid_a)

                fock_a = np.zeros_like(np.asarray(fock.alpha["+-"]))
                for num, c in enumerate(ci_a[:-1]):
                    fock_a += c * np.asarray(fock_list[num].alpha["+-"])

                B_b = np.empty((diis_count + 1, diis_count + 1))
                B_b[-1, :] = -1
                B_b[:, -1] = -1
                B_b[-1, -1] = 0
                for num1, e1 in enumerate(diis_error):
                    for num2, e2 in enumerate(diis_error):
                        if num2 > num1:
                            continue
                        val = np.einsum("ij,ij->", e1.beta["+-"], e2.beta["+-"])
                        B_b[num1, num2] = val
                        B_b[num2, num1] = val

                B_b[:-1, :-1] /= np.abs(B_b[:-1, :-1]).max()

                resid_b = np.zeros(diis_count + 1)
                resid_b[-1] = -1

                ci_b = np.linalg.solve(B_b, resid_b)

                fock_b = np.zeros_like(np.asarray(fock.beta["+-"]))
                for num, c in enumerate(ci_b[:-1]):
                    fock_b += c * np.asarray(fock_list[num].beta["+-"])

                fock = ElectronicIntegrals.from_raw_integrals(fock_a, h1_b=fock_b)

            if scf_iter == max_iter:
                raise Exception("Maximum number of SCF iterations exceeded.")

        # NOTE: from now on fragment_a will no longer change!

        # Post iterations
        logger.info("\nSCF converged.")
        logger.info("Final SCF A-in-B Energy: %s [Eh]", e_new_a)

        return fock, density_a, fragment_a, e_new_a

    def _orthogonalize(self, mo_coeff_vir_ints, mo_coeff_projected, fock):
        if (
            np.linalg.norm(mo_coeff_projected.alpha["+-"]) < 1e-05
            and np.linalg.norm(mo_coeff_projected.beta["+-"]) < 1e-05
        ):
            logger.info("occupied and unoccupied are orthogonal")
            return mo_coeff_vir_ints

        logger.info("occupied and unoccupied are NOT orthogonal")

        # orthogonalization procedure
        # method related to projected atomic orbitals (PAOs)
        # P. Pulay, Chem. Phys. Lett. 100, 151 (1983).
        mo_coeff_vir_projected = mo_coeff_vir_ints - mo_coeff_projected

        einsummed = ElectronicIntegrals.einsum(
            {"ji,jk,kl->il": ("+-",) * 4},
            mo_coeff_vir_projected,
            self.overlap,
            mo_coeff_vir_projected,
            validate=False,
        )

        eigval, eigvec = ElectronicIntegrals.apply(
            np.linalg.eigh, einsummed, multi=True, validate=False
        )
        eigval = ElectronicIntegrals.apply(
            lambda arr: np.linalg.inv(np.diag(np.sqrt(arr))), eigval, validate=False
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

        _, eigvec_fock = ElectronicIntegrals.apply(
            np.linalg.eigh, einsummed, multi=True, validate=False
        )

        mo_coeff_vir_ints = ElectronicIntegrals.einsum(
            {"ij,jk->ik": ("+-",) * 3}, mo_coeff_vir_ints, eigvec_fock, validate=False
        )

        return mo_coeff_vir_ints

    def _fock_build_a(self, density_a: ElectronicDensity, density_b: ElectronicDensity):
        density_tot = density_a + density_b

        h_core = self.hamiltonian.electronic_integrals.one_body

        fock_a = self.hamiltonian.fock(density_a)
        fock_tot = self.hamiltonian.fock(density_tot)

        e_low_level = 0.5 * ElectronicIntegrals.einsum(
            {"ij,ij": ("+-", "+-", "")}, fock_tot + h_core, density_tot
        )
        e_low_level -= 0.5 * ElectronicIntegrals.einsum(
            {"ij,ij": ("+-", "+-", "")}, fock_a + h_core, density_a
        )

        fock_final = self.hamiltonian.fock(density_a)
        fock_final += (fock_tot - h_core) - (fock_a - h_core)

        return fock_final, (
            e_low_level.alpha.get("", 0.0)
            + e_low_level.beta.get("", 0.0)
            + e_low_level.beta_alpha.get("", 0.0)
        )


def _concentric_localization(
    overlap_pb_wb, projection_basis, mo_coeff_vir, num_bf, fock
):
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

    # TODO: make zeta configurable
    zeta = 1
    mo_coeff_vir_cur = mo_coeff_vir_new

    for _ in range(zeta - 1):
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
    logger.info("Pseudocanonicalizing the selected and excluded virtuals separately")

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

    # WARNING: the number of alpha- and beta-spin orbitals is not guaranteed to be identical after
    # this step!
    nvir_a_beta, nvir_b_beta = None, None
    if "+-" in mo_coeff_vir_new.beta:
        nvir_a_beta = mo_coeff_vir_new.beta["+-"].shape[1]
        nvir_b_beta = mo_coeff_vir.beta["+-"].shape[1] - nvir_a_beta

    return mo_coeff_vir, (nvir_a_alpha, nvir_a_beta), (nvir_b_alpha, nvir_b_beta)
