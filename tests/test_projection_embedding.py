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

import numpy as np

import qiskit_nature.optionals as _optionals
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.formats.qcschema_translator import get_ao_to_mo_from_qcschema
from qiskit_nature.second_q.operators import PolynomialTensor
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
        trafo = ProjectionTransformer(2, 1, 0, 0, basis_trafo)
        problem = trafo.transform(problem)

        with self.subTest("alpha coefficients"):
            expected = PolynomialTensor(
                {
                    "+-": np.array([[1.58449216, 0.06134847], [0.06134847, 0.58387659]]),
                    "++--": np.array(
                        [
                            [
                                [[0.78800375, 0.06134847], [0.06134847, 0.18505743]],
                                [[0.06134847, 0.18505743], [0.70806747, 0.02188155]],
                            ],
                            [
                                [[0.06134847, 0.70806747], [0.18505743, 0.02188155]],
                                [[0.18505743, 0.02188155], [0.02188155, 0.69123933]],
                            ],
                        ]
                    ),
                }
            )
            actual = problem.hamiltonian.electronic_integrals.alpha
            self.assertTrue(PolynomialTensor.apply(np.abs, actual).equiv(expected))

        with self.subTest("beta coefficients"):
            self.assertTrue(problem.hamiltonian.electronic_integrals.beta.is_empty())

        with self.subTest("beta_alpha coefficients"):
            self.assertTrue(problem.hamiltonian.electronic_integrals.beta_alpha.is_empty())

        with self.subTest("energy shifts"):
            self.assertEqual(
                problem.hamiltonian.constants.keys(),
                {"nuclear_repulsion_energy", "ProjectionTransformer"},
            )
            self.assertAlmostEqual(
                problem.hamiltonian.constants["ProjectionTransformer"], 2.38098056
            )
            self.assertAlmostEqual(problem.hamiltonian.nuclear_repulsion_energy, -152.1284012)

        with self.subTest("more problem attributes"):
            self.assertEqual(problem.num_spatial_orbitals, 2)
            self.assertEqual(problem.num_alpha, 1)
            self.assertEqual(problem.num_beta, 1)


if __name__ == "__main__":
    unittest.main()
