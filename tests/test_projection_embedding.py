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
from qiskit_nature.second_q.algorithms import GroundStateEigensolver, NumPyMinimumEigensolverFactory
from qiskit_nature.second_q.drivers import MethodType, PySCFDriver
from qiskit_nature.second_q.formats.qcschema_translator import get_ao_to_mo_from_qcschema
from qiskit_nature.second_q.mappers import JordanWignerMapper
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

        # TODO: fix me!
        # with self.subTest("beta coefficients"):
        #     self.assertTrue(problem.hamiltonian.electronic_integrals.beta.is_empty())

        # TODO: fix me!
        # with self.subTest("beta_alpha coefficients"):
        #     self.assertTrue(problem.hamiltonian.electronic_integrals.beta_alpha.is_empty())

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

    # @unittest.skip("debugging")
    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_larger_system(self):
        """Tests a full run through of the transformer."""
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
        )
        driver.run_pyscf()
        qcschema = driver.to_qcschema()
        problem = driver.to_problem(basis=ElectronicBasis.AO, include_dipole=False)
        basis_trafo = get_ao_to_mo_from_qcschema(qcschema)
        trafo = ProjectionTransformer((8, 6), 10, 5, 8, basis_trafo)
        # trafo = ProjectionTransformer(14, 10, 5, 8, basis_trafo)
        problem = trafo.transform(problem)
        print(problem.num_particles)
        print(problem.num_spatial_orbitals)

        # with self.subTest("alpha coefficients"):
        #     prev_atol = PolynomialTensor.atol
        #     prev_rtol = PolynomialTensor.rtol
        #     PolynomialTensor.atol = 1e-6
        #     PolynomialTensor.rtol = 1e-6
        #     expected = PolynomialTensor(
        #         {
        #             "+-": np.array(
        #                 [
        #                     [-2.32871936e00, 8.10672440e-09, -5.10759874e-06, -2.68738034e-01],
        #                     [8.10672439e-09, -2.37706700e00, -2.16582603e-02, 2.66693581e-06],
        #                     [-5.10759874e-06, -2.16582603e-02, -8.28091521e-01, -7.06031308e-06],
        #                     [-2.68738034e-01, 2.66693581e-06, -7.06031308e-06, -1.09946284e00],
        #                 ]
        #             ),
        #             "++--": np.array(
        #                 [
        #                     [
        #                         [
        #                             [
        #                                 6.17224282e-01,
        #                                 -3.50326003e-09,
        #                                 7.04396209e-07,
        #                                 3.17410732e-02,
        #                             ],
        #                             [
        #                                 -3.50326003e-09,
        #                                 3.17301387e-02,
        #                                 -1.47863167e-03,
        #                                 1.82383656e-07,
        #                             ],
        #                             [
        #                                 7.04396209e-07,
        #                                 -1.47863167e-03,
        #                                 8.44276537e-03,
        #                                 1.39166893e-06,
        #                             ],
        #                             [
        #                                 3.17410732e-02,
        #                                 1.82383656e-07,
        #                                 1.39166893e-06,
        #                                 9.49406576e-02,
        #                             ],
        #                         ],
        #                         [
        #                             [
        #                                 -3.50326003e-09,
        #                                 3.17301387e-02,
        #                                 -1.47863167e-03,
        #                                 1.82383656e-07,
        #                             ],
        #                             [
        #                                 5.47736847e-01,
        #                                 -4.60346436e-09,
        #                                 2.44842544e-06,
        #                                 1.32303880e-01,
        #                             ],
        #                             [
        #                                 4.67009984e-03,
        #                                 4.93648350e-07,
        #                                 5.54851667e-08,
        #                                 2.69842793e-03,
        #                             ],
        #                             [
        #                                 -5.58187736e-07,
        #                                 2.76107990e-02,
        #                                 4.72957291e-04,
        #                                 -4.29729858e-07,
        #                             ],
        #                         ],
        #                         [
        #                             [
        #                                 7.04396209e-07,
        #                                 -1.47863167e-03,
        #                                 8.44276537e-03,
        #                                 1.39166893e-06,
        #                             ],
        #                             [
        #                                 4.67009984e-03,
        #                                 4.93648350e-07,
        #                                 5.54851667e-08,
        #                                 2.69842793e-03,
        #                             ],
        #                             [
        #                                 3.52200956e-01,
        #                                 1.99008140e-08,
        #                                 -2.70240928e-07,
        #                                 -1.28046309e-02,
        #                             ],
        #                             [
        #                                 1.83163327e-06,
        #                                 3.68359952e-04,
        #                                 -1.46926024e-03,
        #                                 1.17013751e-06,
        #                             ],
        #                         ],
        #                         [
        #                             [
        #                                 3.17410732e-02,
        #                                 1.82383656e-07,
        #                                 1.39166893e-06,
        #                                 9.49406576e-02,
        #                             ],
        #                             [
        #                                 -5.58187736e-07,
        #                                 2.76107990e-02,
        #                                 4.72957291e-04,
        #                                 -4.29729858e-07,
        #                             ],
        #                             [
        #                                 1.83163327e-06,
        #                                 3.68359952e-04,
        #                                 -1.46926024e-03,
        #                                 1.17013751e-06,
        #                             ],
        #                             [
        #                                 4.59068308e-01,
        #                                 -1.10960792e-07,
        #                                 5.44245322e-07,
        #                                 3.60641330e-03,
        #                             ],
        #                         ],
        #                     ],
        #                     [
        #                         [
        #                             [
        #                                 -3.50326003e-09,
        #                                 5.47736847e-01,
        #                                 4.67009984e-03,
        #                                 -5.58187736e-07,
        #                             ],
        #                             [
        #                                 3.17301387e-02,
        #                                 -4.60346436e-09,
        #                                 4.93648350e-07,
        #                                 2.76107990e-02,
        #                             ],
        #                             [
        #                                 -1.47863167e-03,
        #                                 2.44842544e-06,
        #                                 5.54851667e-08,
        #                                 4.72957291e-04,
        #                             ],
        #                             [
        #                                 1.82383656e-07,
        #                                 1.32303880e-01,
        #                                 2.69842793e-03,
        #                                 -4.29729858e-07,
        #                             ],
        #                         ],
        #                         [
        #                             [
        #                                 3.17301387e-02,
        #                                 -4.60346436e-09,
        #                                 4.93648350e-07,
        #                                 2.76107990e-02,
        #                             ],
        #                             [
        #                                 -4.60346436e-09,
        #                                 8.44293787e-01,
        #                                 1.08394289e-02,
        #                                 -1.36817668e-06,
        #                             ],
        #                             [
        #                                 4.93648350e-07,
        #                                 1.08394289e-02,
        #                                 2.09494731e-02,
        #                                 4.78123721e-07,
        #                             ],
        #                             [
        #                                 2.76107990e-02,
        #                                 -1.36817668e-06,
        #                                 4.78123721e-07,
        #                                 3.42553464e-02,
        #                             ],
        #                         ],
        #                         [
        #                             [
        #                                 -1.47863167e-03,
        #                                 2.44842544e-06,
        #                                 5.54851667e-08,
        #                                 4.72957291e-04,
        #                             ],
        #                             [
        #                                 4.93648350e-07,
        #                                 1.08394289e-02,
        #                                 2.09494731e-02,
        #                                 4.78123721e-07,
        #                             ],
        #                             [
        #                                 1.99008140e-08,
        #                                 3.29292492e-01,
        #                                 -3.00956368e-03,
        #                                 -5.33813273e-08,
        #                             ],
        #                             [
        #                                 3.68359952e-04,
        #                                 2.63341960e-06,
        #                                 1.95356290e-07,
        #                                 -2.59808606e-03,
        #                             ],
        #                         ],
        #                         [
        #                             [
        #                                 1.82383656e-07,
        #                                 1.32303880e-01,
        #                                 2.69842793e-03,
        #                                 -4.29729858e-07,
        #                             ],
        #                             [
        #                                 2.76107990e-02,
        #                                 -1.36817668e-06,
        #                                 4.78123721e-07,
        #                                 3.42553464e-02,
        #                             ],
        #                             [
        #                                 3.68359952e-04,
        #                                 2.63341960e-06,
        #                                 1.95356290e-07,
        #                                 -2.59808606e-03,
        #                             ],
        #                             [
        #                                 -1.10960792e-07,
        #                                 4.25382124e-01,
        #                                 1.60230695e-03,
        #                                 4.66531678e-07,
        #                             ],
        #                         ],
        #                     ],
        #                     [
        #                         [
        #                             [
        #                                 7.04396209e-07,
        #                                 4.67009984e-03,
        #                                 3.52200956e-01,
        #                                 1.83163327e-06,
        #                             ],
        #                             [
        #                                 -1.47863167e-03,
        #                                 4.93648350e-07,
        #                                 1.99008140e-08,
        #                                 3.68359952e-04,
        #                             ],
        #                             [
        #                                 8.44276537e-03,
        #                                 5.54851667e-08,
        #                                 -2.70240928e-07,
        #                                 -1.46926024e-03,
        #                             ],
        #                             [
        #                                 1.39166893e-06,
        #                                 2.69842793e-03,
        #                                 -1.28046309e-02,
        #                                 1.17013751e-06,
        #                             ],
        #                         ],
        #                         [
        #                             [
        #                                 -1.47863167e-03,
        #                                 4.93648350e-07,
        #                                 1.99008140e-08,
        #                                 3.68359952e-04,
        #                             ],
        #                             [
        #                                 2.44842544e-06,
        #                                 1.08394289e-02,
        #                                 3.29292492e-01,
        #                                 2.63341960e-06,
        #                             ],
        #                             [
        #                                 5.54851667e-08,
        #                                 2.09494731e-02,
        #                                 -3.00956368e-03,
        #                                 1.95356290e-07,
        #                             ],
        #                             [
        #                                 4.72957291e-04,
        #                                 4.78123721e-07,
        #                                 -5.33813273e-08,
        #                                 -2.59808606e-03,
        #                             ],
        #                         ],
        #                         [
        #                             [
        #                                 8.44276537e-03,
        #                                 5.54851667e-08,
        #                                 -2.70240928e-07,
        #                                 -1.46926024e-03,
        #                             ],
        #                             [
        #                                 5.54851667e-08,
        #                                 2.09494731e-02,
        #                                 -3.00956368e-03,
        #                                 1.95356290e-07,
        #                             ],
        #                             [
        #                                 -2.70240928e-07,
        #                                 -3.00956368e-03,
        #                                 3.82634543e-01,
        #                                 -5.15974415e-07,
        #                             ],
        #                             [
        #                                 -1.46926024e-03,
        #                                 1.95356290e-07,
        #                                 -5.15974415e-07,
        #                                 1.43934551e-02,
        #                             ],
        #                         ],
        #                         [
        #                             [
        #                                 1.39166893e-06,
        #                                 2.69842793e-03,
        #                                 -1.28046309e-02,
        #                                 1.17013751e-06,
        #                             ],
        #                             [
        #                                 4.72957291e-04,
        #                                 4.78123721e-07,
        #                                 -5.33813273e-08,
        #                                 -2.59808606e-03,
        #                             ],
        #                             [
        #                                 -1.46926024e-03,
        #                                 1.95356290e-07,
        #                                 -5.15974415e-07,
        #                                 1.43934551e-02,
        #                             ],
        #                             [
        #                                 5.44245322e-07,
        #                                 1.60230695e-03,
        #                                 3.48438024e-01,
        #                                 7.30049115e-07,
        #                             ],
        #                         ],
        #                     ],
        #                     [
        #                         [
        #                             [
        #                                 3.17410732e-02,
        #                                 -5.58187736e-07,
        #                                 1.83163327e-06,
        #                                 4.59068308e-01,
        #                             ],
        #                             [
        #                                 1.82383656e-07,
        #                                 2.76107990e-02,
        #                                 3.68359952e-04,
        #                                 -1.10960792e-07,
        #                             ],
        #                             [
        #                                 1.39166893e-06,
        #                                 4.72957291e-04,
        #                                 -1.46926024e-03,
        #                                 5.44245322e-07,
        #                             ],
        #                             [
        #                                 9.49406576e-02,
        #                                 -4.29729858e-07,
        #                                 1.17013751e-06,
        #                                 3.60641330e-03,
        #                             ],
        #                         ],
        #                         [
        #                             [
        #                                 1.82383656e-07,
        #                                 2.76107990e-02,
        #                                 3.68359952e-04,
        #                                 -1.10960792e-07,
        #                             ],
        #                             [
        #                                 1.32303880e-01,
        #                                 -1.36817668e-06,
        #                                 2.63341960e-06,
        #                                 4.25382124e-01,
        #                             ],
        #                             [
        #                                 2.69842793e-03,
        #                                 4.78123721e-07,
        #                                 1.95356290e-07,
        #                                 1.60230695e-03,
        #                             ],
        #                             [
        #                                 -4.29729858e-07,
        #                                 3.42553464e-02,
        #                                 -2.59808606e-03,
        #                                 4.66531678e-07,
        #                             ],
        #                         ],
        #                         [
        #                             [
        #                                 1.39166893e-06,
        #                                 4.72957291e-04,
        #                                 -1.46926024e-03,
        #                                 5.44245322e-07,
        #                             ],
        #                             [
        #                                 2.69842793e-03,
        #                                 4.78123721e-07,
        #                                 1.95356290e-07,
        #                                 1.60230695e-03,
        #                             ],
        #                             [
        #                                 -1.28046309e-02,
        #                                 -5.33813273e-08,
        #                                 -5.15974415e-07,
        #                                 3.48438024e-01,
        #                             ],
        #                             [
        #                                 1.17013751e-06,
        #                                 -2.59808606e-03,
        #                                 1.43934551e-02,
        #                                 7.30049115e-07,
        #                             ],
        #                         ],
        #                         [
        #                             [
        #                                 9.49406576e-02,
        #                                 -4.29729858e-07,
        #                                 1.17013751e-06,
        #                                 3.60641330e-03,
        #                             ],
        #                             [
        #                                 -4.29729858e-07,
        #                                 3.42553464e-02,
        #                                 -2.59808606e-03,
        #                                 4.66531678e-07,
        #                             ],
        #                             [
        #                                 1.17013751e-06,
        #                                 -2.59808606e-03,
        #                                 1.43934551e-02,
        #                                 7.30049115e-07,
        #                             ],
        #                             [
        #                                 3.60641330e-03,
        #                                 4.66531678e-07,
        #                                 7.30049115e-07,
        #                                 4.26386945e-01,
        #                             ],
        #                         ],
        #                     ],
        #                 ]
        #             ),
        #         }
        #     )
        #     actual = problem.hamiltonian.electronic_integrals.one_body.alpha
        #     print(actual)
        #     try:
        #         self.assertTrue(
        #             PolynomialTensor.apply(np.abs, actual).equiv(
        #                 PolynomialTensor.apply(np.abs, expected)
        #             )
        #         )
        #     finally:
        #         PolynomialTensor.atol = prev_atol
        #         PolynomialTensor.rtol = prev_rtol

        # TODO: fix me!
        # with self.subTest("beta coefficients"):
        #     self.assertTrue(problem.hamiltonian.electronic_integrals.beta.is_empty())

        # TODO: fix me!
        # with self.subTest("beta_alpha coefficients"):
        #     self.assertTrue(problem.hamiltonian.electronic_integrals.beta_alpha.is_empty())

        # with self.subTest("energy shifts"):
        #     self.assertEqual(
        #         problem.hamiltonian.constants.keys(),
        #         {"nuclear_repulsion_energy", "ProjectionTransformer"},
        #     )
        #     self.assertAlmostEqual(
        #         problem.hamiltonian.constants["ProjectionTransformer"], 5.822567531331251
        #     )
        #     self.assertAlmostEqual(problem.hamiltonian.nuclear_repulsion_energy, -206.5952584223863)

        # with self.subTest("more problem attributes"):
        #     self.assertEqual(problem.num_spatial_orbitals, 4)
        #     self.assertEqual(problem.num_alpha, 2)
        #     self.assertEqual(problem.num_beta, 2)

        algo = GroundStateEigensolver(JordanWignerMapper(), NumPyMinimumEigensolverFactory())

        res = algo.solve(problem)
        print(res)

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
        trafo = ProjectionTransformer(2, 1, 0, 0, basis_trafo)
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
        driver = PySCFDriver(
            atom="N 0.0 0.0 0.0; N 0.0 0.0 1.2",
            # method=MethodType.UHF,
        )
        driver.run_pyscf()
        qcschema = driver.to_qcschema()
        problem = driver.to_problem(basis=ElectronicBasis.AO, include_dipole=False)
        print(problem.num_particles)
        basis_trafo = get_ao_to_mo_from_qcschema(qcschema)
        trafo = ProjectionTransformer(4, 5, 0, 0, basis_trafo)
        problem = trafo.transform(problem)
        print(problem.num_particles)

        # with self.subTest("energy shifts"):
        #     self.assertEqual(
        #         problem.hamiltonian.constants.keys(),
        #         {"nuclear_repulsion_energy", "ProjectionTransformer"},
        #     )
        #     self.assertAlmostEqual(
        #         problem.hamiltonian.constants["ProjectionTransformer"], 2.380961985
        #     )
        #     self.assertAlmostEqual(problem.hamiltonian.nuclear_repulsion_energy, -152.1284012)

        # with self.subTest("more problem attributes"):
        #     self.assertEqual(problem.num_spatial_orbitals, 4)
        #     self.assertEqual(problem.num_alpha, 2)
        #     self.assertEqual(problem.num_beta, 2)

        algo = GroundStateEigensolver(JordanWignerMapper(), NumPyMinimumEigensolverFactory())

        res = algo.solve(problem)
        print(res)

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
        print(problem.num_particles)
        basis_trafo = get_ao_to_mo_from_qcschema(qcschema)
        trafo = ProjectionTransformer((3, 1), 5, 0, 0, basis_trafo)
        problem = trafo.transform(problem)
        print(problem.num_particles)

        # with self.subTest("energy shifts"):
        #     self.assertEqual(
        #         problem.hamiltonian.constants.keys(),
        #         {"nuclear_repulsion_energy", "ProjectionTransformer"},
        #     )
        #     self.assertAlmostEqual(
        #         problem.hamiltonian.constants["ProjectionTransformer"], 2.380961985
        #     )
        #     self.assertAlmostEqual(problem.hamiltonian.nuclear_repulsion_energy, -152.1284012)

        # with self.subTest("more problem attributes"):
        #     self.assertEqual(problem.num_spatial_orbitals, 4)
        #     self.assertEqual(problem.num_alpha, 2)
        #     self.assertEqual(problem.num_beta, 2)

        algo = GroundStateEigensolver(JordanWignerMapper(), NumPyMinimumEigensolverFactory())

        res = algo.solve(problem)
        print(res)


if __name__ == "__main__":
    unittest.main()
