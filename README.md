
# Spinsolveproc


<div align="center">

[![PyPI - Version](https://img.shields.io/pypi/v/spinsolveproc.svg)](https://pypi.python.org/pypi/spinsolveproc)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/spinsolveproc.svg)](https://pypi.python.org/pypi/spinsolveproc)
[![Tests](https://github.com/rserial/spinsolveproc/workflows/tests/badge.svg)](https://github.com/rserial/spinsolveproc/actions?workflow=tests)
[![Codecov](https://codecov.io/gh/rserial/spinsolveproc/branch/main/graph/badge.svg)](https://codecov.io/gh/rserial/spinsolveproc)
[![Read the Docs](https://readthedocs.org/projects/spinsolveproc/badge/)](https://spinsolveproc.readthedocs.io/)
[![PyPI - License](https://img.shields.io/pypi/l/spinsolveproc.svg)](https://pypi.python.org/pypi/spinsolveproc)

[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](https://www.contributor-covenant.org/version/2/0/code_of_conduct/)

</div>


Python library to process Spinsolve NMR experiments. The library can recognise both standard and expert softward files.

* GitHub repo: <https://github.com/rserial/spinsolveproc.git>
<!-- * Documentation: <https://spinsolveproc.readthedocs.io> -->
* Free software: GNU General Public License v3

## Features
The library currently supports the following experiments:

[x] Proton: Computes FID spectra and exports obtained data to `./processed data`.

[x] T2: Finds the peaks in the spectra and calculates the T2 decay associated to each peak. It exports the obtained data to `./processed data`. 

[x] T2Bulk: Constructs T2 decay array and performs a monoexponential fitting. It exports the obtained data to `./processed data`.

[x] T1: Finds the peaks in the spectra and calculates the T1 decay associated to each peak. It exports the obtained data to `./processed data`.

[x] T1IRT2


## Quickstart

```
pip install git+https://github.com/rserial/spinsolveproc.git
```

## Usage from console

```
spinsolveproc process_exp [dir] [options]
```
- dir: The parent directory containing all experiment directories.

- options:
    - `--all`: Process all experiments in the directory.
    - `experiment_name`: Specify the experiment name (`Proton`, `T2`, `T2Bulk`,`T1`, `T1IRT2`).

## Credits

This package was created with [Cookiecutter][cookiecutter] and the [fedejaure/cookiecutter-modern-pypackage][cookiecutter-modern-pypackage] project template.

[cookiecutter]: https://github.com/cookiecutter/cookiecutter
[cookiecutter-modern-pypackage]: https://github.com/fedejaure/cookiecutter-modern-pypackage
