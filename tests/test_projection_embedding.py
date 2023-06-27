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

"""Tests for the ProjectionTransformer."""

import unittest
from functools import partial
from test import QiskitNatureTestCase

import numpy as np
from qiskit.test import slow_test

import qiskit_nature.optionals as _optionals
from qiskit_nature.second_q.drivers import MethodType, PySCFDriver
from qiskit_nature.second_q.formats.qcschema_translator import \
    get_ao_to_mo_from_qcschema
from qiskit_nature.second_q.operators import ElectronicIntegrals
from qiskit_nature.second_q.problems import ElectronicBasis
from qiskit_nature.second_q.transformers import ProjectionTransformer


class TestProjectionTransformer(QiskitNatureTestCase):
    """ProjectionTransformer tests."""

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_full_run(self):
        """Tests a full run through of the transformer."""
        driver = PySCFDriver(
            atom="""H -1.9237 0.3850 0.0000;
                O -1.1867 -0.2472 0.0000;
                H -0.0227 1.1812 0.8852;
                C 0.0000 0.5526 0.0000;
                H -0.0227 1.1812 -0.8852;
                C 1.1879 -0.3829 0.0000;
                H 2.0985 0.2306 0.0000;
                H 1.1184 -1.0093 0.8869;
                H 1.1184 -1.0093 -0.8869"""
        )
        driver.run_pyscf()
        qcschema = driver.to_qcschema()
        problem = driver.to_problem(basis=ElectronicBasis.AO, include_dipole=False)
        basis_trafo = get_ao_to_mo_from_qcschema(qcschema)
        trafo = ProjectionTransformer(2, 1, basis_trafo)
        problem = trafo.transform(problem)

        with self.subTest("energy shifts"):
            self.assertEqual(
                problem.hamiltonian.constants.keys(),
                {"nuclear_repulsion_energy", "ProjectionTransformer"},
            )
            self.assertAlmostEqual(
                problem.hamiltonian.constants["ProjectionTransformer"], 2.38098439
            )
            self.assertAlmostEqual(problem.hamiltonian.nuclear_repulsion_energy, -152.1284012)

        with self.subTest("more problem attributes"):
            self.assertEqual(problem.num_spatial_orbitals, 2)
            self.assertEqual(problem.num_alpha, 1)
            self.assertEqual(problem.num_beta, 1)

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_unrestricted(self):
        """Tests the unrestricted spin case."""
        driver = PySCFDriver(
            atom="""H -1.9237 0.3850 0.0000;
                O -1.1867 -0.2472 0.0000;
                H -0.0227 1.1812 0.8852;
                C 0.0000 0.5526 0.0000;
                H -0.0227 1.1812 -0.8852;
                C 1.1879 -0.3829 0.0000;
                H 2.0985 0.2306 0.0000;
                H 1.1184 -1.0093 0.8869;
                H 1.1184 -1.0093 -0.8869""",
            method=MethodType.UHF,
        )
        driver.run_pyscf()
        qcschema = driver.to_qcschema()
        problem = driver.to_problem(basis=ElectronicBasis.AO, include_dipole=False)
        basis_trafo = get_ao_to_mo_from_qcschema(qcschema)
        trafo = ProjectionTransformer(2, 1, basis_trafo)
        problem = trafo.transform(problem)

        with self.subTest("energy shifts"):
            self.assertEqual(
                problem.hamiltonian.constants.keys(),
                {"nuclear_repulsion_energy", "ProjectionTransformer"},
            )
            self.assertAlmostEqual(
                problem.hamiltonian.constants["ProjectionTransformer"], 2.380961985
            )
            self.assertAlmostEqual(problem.hamiltonian.nuclear_repulsion_energy, -152.1284012)

        with self.subTest("more problem attributes"):
            self.assertEqual(problem.num_spatial_orbitals, 2)
            self.assertEqual(problem.num_alpha, 1)
            self.assertEqual(problem.num_beta, 1)

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_n2(self):
        """TODO."""
        driver = PySCFDriver(atom="N 0.0 0.0 0.0; N 0.0 0.0 1.2")
        driver.run_pyscf()
        qcschema = driver.to_qcschema()
        problem = driver.to_problem(basis=ElectronicBasis.AO, include_dipole=False)
        basis_trafo = get_ao_to_mo_from_qcschema(qcschema)
        trafo = ProjectionTransformer(4, 5, basis_trafo, 4, 4)
        problem = trafo.transform(problem)

        with self.subTest("energy shifts"):
            self.assertEqual(
                problem.hamiltonian.constants.keys(),
                {"nuclear_repulsion_energy", "ProjectionTransformer"},
            )
            self.assertAlmostEqual(
                problem.hamiltonian.constants["ProjectionTransformer"], 41.108056563
            )
            self.assertAlmostEqual(problem.hamiltonian.nuclear_repulsion_energy, -107.487783928)

        with self.subTest("more problem attributes"):
            self.assertEqual(problem.num_spatial_orbitals, 4)
            self.assertEqual(problem.num_alpha, 2)
            self.assertEqual(problem.num_beta, 2)

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_o2(self):
        """TODO."""
        driver = PySCFDriver(
            atom="O 0.0 0.0 0.0; O 0.0 0.0 1.2",
            spin=2,
            method=MethodType.UHF,
        )
        driver.run_pyscf()
        qcschema = driver.to_qcschema()
        problem = driver.to_problem(basis=ElectronicBasis.AO, include_dipole=False)
        basis_trafo = get_ao_to_mo_from_qcschema(qcschema)
        trafo = ProjectionTransformer((4, 2), 5, basis_trafo)
        problem = trafo.transform(problem)

        with self.subTest("energy shifts"):
            self.assertEqual(
                problem.hamiltonian.constants.keys(),
                {"nuclear_repulsion_energy", "ProjectionTransformer"},
            )
            self.assertAlmostEqual(
                problem.hamiltonian.constants["ProjectionTransformer"], 62.082773142
            )
            self.assertAlmostEqual(problem.hamiltonian.nuclear_repulsion_energy, -147.633452733)

        with self.subTest("more problem attributes"):
            self.assertEqual(problem.num_spatial_orbitals, 5)
            self.assertEqual(problem.num_alpha, 4)
            self.assertEqual(problem.num_beta, 2)

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_larger_system(self):
        """Tests a full run on a larger system."""
        driver = PySCFDriver(
            atom="""N          2.54840       -0.23120       -0.00000;
                C          1.79831        0.04694       -0.00000;
                C          0.37007        0.56616       -0.00000;
                H          0.22595        1.20906       -0.90096;
                H          0.22596        1.20907        0.90095;
                C         -0.68181       -0.60624        0.00001;
                H         -0.51758       -1.24375       -0.89892;
                H         -0.51758       -1.24374        0.89895;
                C         -2.14801       -0.05755        0.00001;
                H         -2.87394       -0.89898        0.00001;
                H         -2.33417        0.56585        0.90131;
                H         -2.33417        0.56584       -0.90131""",
            basis="sto3g",
        )
        driver.run_pyscf()
        qcschema = driver.to_qcschema()
        problem = driver.to_problem(basis=ElectronicBasis.AO, include_dipole=False)
        basis_trafo = get_ao_to_mo_from_qcschema(qcschema)
        trafo = ProjectionTransformer(14, 10, basis_trafo, 4, 4)
        problem = trafo.transform(problem)

        with self.subTest("energy shifts"):
            self.assertEqual(
                problem.hamiltonian.constants.keys(),
                {"nuclear_repulsion_energy", "ProjectionTransformer"},
            )
            self.assertAlmostEqual(
                problem.hamiltonian.constants["ProjectionTransformer"], 5.822567531
            )
            self.assertAlmostEqual(problem.hamiltonian.nuclear_repulsion_energy, -206.595258422)

        with self.subTest("more problem attributes"):
            self.assertEqual(problem.num_spatial_orbitals, 4)
            self.assertEqual(problem.num_alpha, 2)
            self.assertEqual(problem.num_beta, 2)

    @slow_test
    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_larger_system_uhf(self):
        """Tests a full run on a larger system using UHF."""
        driver = PySCFDriver(
            atom="""N          2.54840       -0.23120       -0.00000;
                C          1.79831        0.04694       -0.00000;
                C          0.37007        0.56616       -0.00000;
                H          0.22595        1.20906       -0.90096;
                H          0.22596        1.20907        0.90095;
                C         -0.68181       -0.60624        0.00001;
                H         -0.51758       -1.24375       -0.89892;
                H         -0.51758       -1.24374        0.89895;
                C         -2.14801       -0.05755        0.00001;
                H         -2.87394       -0.89898        0.00001;
                H         -2.33417        0.56585        0.90131;
                H         -2.33417        0.56584       -0.90131""",
            spin=2,
            method=MethodType.UHF,
            basis="631g*",
        )
        driver.run_pyscf()
        qcschema = driver.to_qcschema()
        problem = driver.to_problem(basis=ElectronicBasis.AO, include_dipole=False)
        basis_trafo = get_ao_to_mo_from_qcschema(qcschema)
        trafo = ProjectionTransformer((8, 6), 28, basis_trafo, (3, 1), 4)
        problem = trafo.transform(problem)

        with self.subTest("energy shifts"):
            self.assertEqual(
                problem.hamiltonian.constants.keys(),
                {"nuclear_repulsion_energy", "ProjectionTransformer"},
            )
            self.assertAlmostEqual(
                problem.hamiltonian.constants["ProjectionTransformer"], 5.366528176
            )
            self.assertAlmostEqual(problem.hamiltonian.nuclear_repulsion_energy, -208.900041579)

        with self.subTest("more problem attributes"):
            self.assertEqual(problem.num_spatial_orbitals, 4)
            self.assertEqual(problem.num_alpha, 3)
            self.assertEqual(problem.num_beta, 1)

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_larger_system_dft(self):
        """Tests a full run on a larger system."""
        driver = PySCFDriver(
            atom="""N          2.54840       -0.23120       -0.00000;
                C          1.79831        0.04694       -0.00000;
                C          0.37007        0.56616       -0.00000;
                H          0.22595        1.20906       -0.90096;
                H          0.22596        1.20907        0.90095;
                C         -0.68181       -0.60624        0.00001;
                H         -0.51758       -1.24375       -0.89892;
                H         -0.51758       -1.24374        0.89895;
                C         -2.14801       -0.05755        0.00001;
                H         -2.87394       -0.89898        0.00001;
                H         -2.33417        0.56585        0.90131;
                H         -2.33417        0.56584       -0.90131""",
            basis="sto3g",
            xc_functional="pbe0",
            method=MethodType.RKS,
        )
        driver.run_pyscf()
        qcschema = driver.to_qcschema()
        problem = driver.to_problem(basis=ElectronicBasis.AO, include_dipole=False)
        basis_trafo = get_ao_to_mo_from_qcschema(qcschema)
        trafo = ProjectionTransformer(14, 10, basis_trafo, 4, 4)

        def _fock_build_a(trafo, density_a, density_b):
            density_tot = density_a + density_b

            h_core = trafo.hamiltonian.electronic_integrals.one_body

            pyscf_rho_a = np.asarray(density_a.trace_spin()["+-"])
            pyscf_rho_tot = np.asarray(density_tot.trace_spin()["+-"])

            pyscf_fock_a = driver._calc.get_fock(dm=pyscf_rho_a)
            pyscf_fock_tot = driver._calc.get_fock(dm=pyscf_rho_tot)

            e_low_level_a = driver._calc.energy_tot(dm=pyscf_rho_a)
            e_low_level_tot = driver._calc.energy_tot(dm=pyscf_rho_tot)

            fock_final = trafo.hamiltonian.fock(density_a)
            h_core_a = h_core.alpha["+-"]
            fock_delta = (pyscf_fock_tot - h_core_a) - (pyscf_fock_a - h_core_a)
            fock_final = ElectronicIntegrals.from_raw_integrals(fock_final.alpha["+-"] + fock_delta)

            e_tot = e_low_level_tot - e_low_level_a

            return fock_final, e_tot

        trafo._fock_build_a = partial(_fock_build_a, trafo)

        problem = trafo.transform(problem)

        with self.subTest("energy shifts"):
            self.assertEqual(
                problem.hamiltonian.constants.keys(),
                {"nuclear_repulsion_energy", "ProjectionTransformer"},
            )
            self.assertAlmostEqual(
                problem.hamiltonian.constants["ProjectionTransformer"], 5.8052028
            )
            self.assertAlmostEqual(problem.hamiltonian.nuclear_repulsion_energy, -207.264214689)

        with self.subTest("more problem attributes"):
            self.assertEqual(problem.num_spatial_orbitals, 4)
            self.assertEqual(problem.num_alpha, 2)
            self.assertEqual(problem.num_beta, 2)

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_larger_system_dft_pm(self):
        """Tests a full run on a larger system."""
        driver = PySCFDriver(
            atom="""N          2.54840       -0.23120       -0.00000;
                C          1.79831        0.04694       -0.00000;
                C          0.37007        0.56616       -0.00000;
                H          0.22595        1.20906       -0.90096;
                H          0.22596        1.20907        0.90095;
                C         -0.68181       -0.60624        0.00001;
                H         -0.51758       -1.24375       -0.89892;
                H         -0.51758       -1.24374        0.89895;
                C         -2.14801       -0.05755        0.00001;
                H         -2.87394       -0.89898        0.00001;
                H         -2.33417        0.56585        0.90131;
                H         -2.33417        0.56584       -0.90131""",
            basis="sto3g",
            xc_functional="pbe",
            method=MethodType.RKS,
        )
        driver.run_pyscf()
        qcschema = driver.to_qcschema()
        problem = driver.to_problem(basis=ElectronicBasis.AO, include_dipole=False)
        print(problem.reference_energy)
        basis_trafo = get_ao_to_mo_from_qcschema(qcschema)
        trafo = ProjectionTransformer(14, 10, basis_trafo, 4, 4)

        def _pipek_mezey(trafo, overlap, mo_coeff_occ, num_bf, nocc_a):
            from pyscf.lo import PipekMezey

            nocc_a = nocc_a[0]
            nocc_b = (driver._mol.nelectron - 2 * nocc_a) // 2

            pm = PipekMezey(driver._mol)
            mo = pm.kernel(mo_coeff_occ.alpha["+-"], verbose=4)

            nocc = mo.shape[1]
            pop = np.zeros((nocc, 2))
            for i in range(nocc):
                col = mo[:, i]
                dens = np.outer(col, col)
                PS = np.dot(dens, overlap)

                pop[i,0] = np.trace(PS[:num_bf, :num_bf])
                pop[i,1] = np.trace(PS[num_bf:, num_bf:])

            print(nocc, nocc_a, nocc_b, num_bf)

            pop_order_1 = np.argsort(-1 * pop[:,0])
            pop_order_2 = np.argsort(-1 * pop[:,1])

            orbid_1 = pop_order_1[:nocc_a]
            orbid_2 = pop_order_2[:nocc_b]

            print("orbitals assigned to fragment 1:", orbid_1)
            print("orbitals assigned to fragment 2:", orbid_2)

            nao = driver._mol.nao
            fragment_1 = np.zeros((nao, nocc_a))
            fragment_2 = np.zeros((nao, nocc_b))

            for i in range(nocc_a):
                fragment_1[:,i] = mo[:,orbid_1[i]]
            for i in range(nocc_b):
                fragment_2[:,i] = mo[:,orbid_2[i]]

            return (
                ElectronicIntegrals.from_raw_integrals(fragment_1, validate=False),
                ElectronicIntegrals.from_raw_integrals(fragment_2, validate=False),
            )

        def _fock_build_a(trafo, density_a, density_b):
            density_tot = density_a + density_b

            h_core = trafo.hamiltonian.electronic_integrals.one_body

            pyscf_rho_a = np.asarray(density_a.trace_spin()["+-"])
            pyscf_rho_tot = np.asarray(density_tot.trace_spin()["+-"])

            pyscf_fock_a = driver._calc.get_fock(dm=pyscf_rho_a)
            pyscf_fock_tot = driver._calc.get_fock(dm=pyscf_rho_tot)

            e_low_level_a = driver._calc.energy_tot(dm=pyscf_rho_a)
            e_low_level_tot = driver._calc.energy_tot(dm=pyscf_rho_tot)

            fock_final = trafo.hamiltonian.fock(density_a)
            h_core_a = h_core.alpha["+-"]
            fock_delta = (pyscf_fock_tot - h_core_a) - (pyscf_fock_a - h_core_a)
            fock_final = ElectronicIntegrals.from_raw_integrals(fock_final.alpha["+-"] + fock_delta)

            e_tot = e_low_level_tot - e_low_level_a

            return fock_final, e_tot

        trafo._spade_partition = partial(_pipek_mezey, trafo)
        trafo._fock_build_a = partial(_fock_build_a, trafo)

        problem = trafo.transform(problem)

        with self.subTest("energy shifts"):
            self.assertEqual(
                problem.hamiltonian.constants.keys(),
                {"nuclear_repulsion_energy", "ProjectionTransformer"},
            )
            self.assertAlmostEqual(
                problem.hamiltonian.constants["ProjectionTransformer"], 5.797826
            )
            self.assertAlmostEqual(problem.hamiltonian.nuclear_repulsion_energy, -207.22327435)

        with self.subTest("more problem attributes"):
            self.assertEqual(problem.num_spatial_orbitals, 4)
            self.assertEqual(problem.num_alpha, 2)
            self.assertEqual(problem.num_beta, 2)


if __name__ == "__main__":
    unittest.main()
