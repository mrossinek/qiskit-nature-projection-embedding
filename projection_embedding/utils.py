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

"""The Projection Embedding."""

from __future__ import annotations

from typing import Sequence

from qiskit_nature.second_q.operators import ElectronicIntegrals


def split_elec_ints_per_spin(
    integrals: ElectronicIntegrals,
    splitting_func: Callable,
    alpha_indices: int | Sequence[int],
    beta_indices: int | Sequence[int],
) -> tuple[ElectronicIntegrals, ElectronicIntegrals]:
    """TODO."""
    left_a, right_a = integrals.alpha.split(
        splitting_func, alpha_indices, validate=False
    )
    left_b, right_b = None, None
    if not integrals.beta.is_empty():
        left_b, right_b = integrals.beta.split(
            splitting_func, beta_indices, validate=False
        )
    return (
        ElectronicIntegrals(left_a, left_b, validate=False),
        ElectronicIntegrals(right_a, right_b, validate=False),
    )
