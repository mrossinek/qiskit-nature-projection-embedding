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

"""Tests for the ProjectionEmbedding."""

import unittest
from functools import partial

import numpy as np
from qiskit.test import slow_test

import qiskit_nature.optionals as _optionals
from qiskit_nature.second_q.drivers import MethodType, PySCFDriver
from qiskit_nature.second_q.formats.qcschema_translator import (
    get_ao_to_mo_from_qcschema,
)
from qiskit_nature.second_q.operators import ElectronicIntegrals
from qiskit_nature.second_q.problems import ElectronicBasis

from projection_embedding.projection_embedding import ProjectionEmbedding
from projection_embedding.occupied_orbital_partitioning import (
    PySCFPipekMezeyPartitioning,
)


class TestProjectionEmbedding(unittest.TestCase):
    """ProjectionEmbedding tests."""

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_full_run(self):
        """Tests a full run through of the embedding."""
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
        overlap = driver._calc.get_ovlp()
        overlap[np.abs(overlap) < 1e-12] = 0.0
        trafo = ProjectionEmbedding(2, 1, basis_trafo, overlap)
        problem = trafo.transform(problem)

        with self.subTest("energy shifts"):
            self.assertEqual(
                problem.hamiltonian.constants.keys(),
                {"nuclear_repulsion_energy", "ProjectionEmbedding"},
            )
            self.assertAlmostEqual(
                problem.hamiltonian.constants["ProjectionEmbedding"],
                2.38098439,
                places=5,
            )
            self.assertAlmostEqual(
                problem.hamiltonian.nuclear_repulsion_energy, -152.1284012, places=5
            )

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
        overlap = driver._calc.get_ovlp()
        overlap[np.abs(overlap) < 1e-12] = 0.0
        trafo = ProjectionEmbedding(2, 1, basis_trafo, overlap)
        problem = trafo.transform(problem)

        with self.subTest("energy shifts"):
            self.assertEqual(
                problem.hamiltonian.constants.keys(),
                {"nuclear_repulsion_energy", "ProjectionEmbedding"},
            )
            self.assertAlmostEqual(
                problem.hamiltonian.constants["ProjectionEmbedding"],
                2.380961985,
                places=5,
            )
            self.assertAlmostEqual(
                problem.hamiltonian.nuclear_repulsion_energy, -152.1284012, places=5
            )

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
        overlap = driver._calc.get_ovlp()
        overlap[np.abs(overlap) < 1e-12] = 0.0
        trafo = ProjectionEmbedding(4, 5, basis_trafo, overlap, 4, 4)
        problem = trafo.transform(problem)

        with self.subTest("energy shifts"):
            self.assertEqual(
                problem.hamiltonian.constants.keys(),
                {"nuclear_repulsion_energy", "ProjectionEmbedding"},
            )
            self.assertAlmostEqual(
                problem.hamiltonian.constants["ProjectionEmbedding"],
                41.108056563,
                places=5,
            )
            self.assertAlmostEqual(
                problem.hamiltonian.nuclear_repulsion_energy, -107.487783928, places=5
            )

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
        overlap = driver._calc.get_ovlp()
        overlap[np.abs(overlap) < 1e-12] = 0.0
        trafo = ProjectionEmbedding((4, 2), 5, basis_trafo, overlap)
        problem = trafo.transform(problem)

        with self.subTest("energy shifts"):
            self.assertEqual(
                problem.hamiltonian.constants.keys(),
                {"nuclear_repulsion_energy", "ProjectionEmbedding"},
            )
            self.assertAlmostEqual(
                problem.hamiltonian.constants["ProjectionEmbedding"],
                62.082773142,
                places=5,
            )
            self.assertAlmostEqual(
                problem.hamiltonian.nuclear_repulsion_energy, -147.633452733, places=5
            )

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
        overlap = driver._calc.get_ovlp()
        overlap[np.abs(overlap) < 1e-12] = 0.0
        trafo = ProjectionEmbedding(14, 10, basis_trafo, overlap, 4, 4)
        problem = trafo.transform(problem)

        with self.subTest("energy shifts"):
            self.assertEqual(
                problem.hamiltonian.constants.keys(),
                {"nuclear_repulsion_energy", "ProjectionEmbedding"},
            )
            self.assertAlmostEqual(
                problem.hamiltonian.constants["ProjectionEmbedding"],
                5.822567531,
                places=5,
            )
            self.assertAlmostEqual(
                problem.hamiltonian.nuclear_repulsion_energy, -206.595258422, places=5
            )

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
        overlap = driver._calc.get_ovlp()
        overlap[np.abs(overlap) < 1e-12] = 0.0
        trafo = ProjectionEmbedding((8, 6), 28, basis_trafo, overlap, (3, 1), 4)
        problem = trafo.transform(problem)

        with self.subTest("energy shifts"):
            self.assertEqual(
                problem.hamiltonian.constants.keys(),
                {"nuclear_repulsion_energy", "ProjectionEmbedding"},
            )
            self.assertAlmostEqual(
                problem.hamiltonian.constants["ProjectionEmbedding"],
                5.366528176,
                places=5,
            )
            self.assertAlmostEqual(
                problem.hamiltonian.nuclear_repulsion_energy, -208.900041579, places=5
            )

        with self.subTest("more problem attributes"):
            self.assertEqual(problem.num_spatial_orbitals, 4)
            self.assertEqual(problem.num_alpha, 3)
            self.assertEqual(problem.num_beta, 1)

    @slow_test
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
        overlap = driver._calc.get_ovlp()
        overlap[np.abs(overlap) < 1e-12] = 0.0
        trafo = ProjectionEmbedding(14, 10, basis_trafo, overlap, 4, 4)

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
            fock_final = ElectronicIntegrals.from_raw_integrals(
                fock_final.alpha["+-"] + fock_delta
            )

            e_tot = e_low_level_tot - e_low_level_a

            return fock_final, e_tot

        trafo._fock_build_a = partial(_fock_build_a, trafo)

        problem = trafo.transform(problem)

        with self.subTest("energy shifts"):
            self.assertEqual(
                problem.hamiltonian.constants.keys(),
                {"nuclear_repulsion_energy", "ProjectionEmbedding"},
            )
            self.assertAlmostEqual(
                problem.hamiltonian.constants["ProjectionEmbedding"],
                5.8052028,
                places=5,
            )
            self.assertAlmostEqual(
                problem.hamiltonian.nuclear_repulsion_energy, -207.264214689, places=5
            )

        with self.subTest("more problem attributes"):
            self.assertEqual(problem.num_spatial_orbitals, 4)
            self.assertEqual(problem.num_alpha, 2)
            self.assertEqual(problem.num_beta, 2)

    @slow_test
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
        basis_trafo = get_ao_to_mo_from_qcschema(qcschema)
        overlap = driver._calc.get_ovlp()
        overlap[np.abs(overlap) < 1e-12] = 0.0
        trafo = ProjectionEmbedding(14, 10, basis_trafo, overlap, 4, 4)

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
            fock_final = ElectronicIntegrals.from_raw_integrals(
                fock_final.alpha["+-"] + fock_delta
            )

            e_tot = e_low_level_tot - e_low_level_a

            return fock_final, e_tot

        trafo.occupied_orbital_partitioning = PySCFPipekMezeyPartitioning(driver._mol)
        trafo._fock_build_a = partial(_fock_build_a, trafo)

        problem = trafo.transform(problem)

        with self.subTest("energy shifts"):
            self.assertEqual(
                problem.hamiltonian.constants.keys(),
                {"nuclear_repulsion_energy", "ProjectionEmbedding"},
            )
            self.assertAlmostEqual(
                problem.hamiltonian.constants["ProjectionEmbedding"],
                5.797826,
                places=3,
            )
            self.assertAlmostEqual(
                problem.hamiltonian.nuclear_repulsion_energy, -207.22327435, places=3
            )

        with self.subTest("more problem attributes"):
            self.assertEqual(problem.num_spatial_orbitals, 4)
            self.assertEqual(problem.num_alpha, 2)
            self.assertEqual(problem.num_beta, 2)


if __name__ == "__main__":
    unittest.main()
