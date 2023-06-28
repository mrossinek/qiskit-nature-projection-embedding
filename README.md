# Projection Embedding using Qiskit Nature

This repository contains the latest prototype implementation of the Projection-based embedding using Qiskit Nature.

## Installation

You can simply install the contents of this repository after cloning it:
```
pip install .
```

:warning: You need to install Qiskit Nature from this branch https://github.com/qiskit-community/qiskit-nature/pull/1215.

## Usage

The file `demo.py` shows an example of how to use this embedding transformer.
After installing, you can run it as:
```
python demo.py
```

## Testing

You can also run the unittests.
```
python -m unittest discover tests
```

## Citing

When using this software, please cite the corresponding paper:

> Max Rossmannek, Fabijan Pavošević, Angel Rubio, Ivano Tavernelli;
> Quantum Embedding Method for the Simulation of Strongly Correlated Systems on Quantum Computers.
> J. Phys. Chem. Lett. 2023, 14, 14, 3491–3497.
>
> https://doi.org/10.1021/acs.jpclett.3c00330

You should also cite [Qiskit](https://github.com/Qiskit/qiskit-terra),
[Qiskit Nature](https://github.com/Qiskit/qiskit-nature) and the classical computational software
which you are using as per the citation instructions provided by each of these software packages.
