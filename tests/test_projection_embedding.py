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
from test import QiskitNatureTestCase

from qiskit.test import slow_test
import qiskit_nature.optionals as _optionals
from qiskit_nature.second_q.drivers import MethodType, PySCFDriver
from qiskit_nature.second_q.formats.qcschema_translator import get_ao_to_mo_from_qcschema
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
            basis="sto3g"
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
            basis="631g*"
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


if __name__ == "__main__":
    unittest.main()
