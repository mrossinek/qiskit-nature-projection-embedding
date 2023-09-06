# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.formats import get_ao_to_mo_from_qcschema
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.problems import ElectronicBasis
from qiskit_nature.settings import settings

from projection_embedding.projection_embedding import ProjectionEmbedding

settings.tensor_unwrapping = False
settings.use_pauli_sum_op = False
settings.use_symmetry_reduced_integrals = True


def _main():
    # setup driver
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

    # setup the AO problem
    driver.run_pyscf()
    qcschema = driver.to_qcschema()
    problem = driver.to_problem(basis=ElectronicBasis.AO, include_dipole=False)

    # extract the basis transformer
    basis_trafo = get_ao_to_mo_from_qcschema(qcschema)

    # extract the AO overlap matrix
    overlap = driver._calc.get_ovlp()
    overlap[np.abs(overlap) < 1e-12] = 0.0

    # setup the projection embedding
    trafo = ProjectionEmbedding(14, 10, basis_trafo, overlap, 4, 4)
    # NOTE: you can also further customize the ProjectionEmbedding instance by varying e.g. the
    #  - fock operator constructor (to support embedding into DFT)
    #  - occupied orbital partitioning (to replace the default SPADE procedure)
    #  - virtual orbital localization

    # transform the problem
    problem = trafo.transform(problem)

    # setup solver
    mapper = JordanWignerMapper()
    solver = NumPyMinimumEigensolver()
    algo = GroundStateEigensolver(mapper, solver)

    # solve the problem
    result = algo.solve(problem)
    print(result)


if __name__ == "__main__":
    _main()
