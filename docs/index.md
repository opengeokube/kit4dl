<div align="center">
<img src="https://raw.githubusercontent.com/opengeokube/ml-kit/56dc56c1be7f6332c0f75cdfb3160d29cebc3c58/static/logo.svg" width="20%" height="20%">
</div>

---
A quick way to start with machine and deep learning
---

[![python](https://img.shields.io/badge/-Python_3.10%7C3.11-blue?logo=python&logoColor=white)](https://www.python.org/downloads)

[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/) 
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![pydocstyle](https://img.shields.io/badge/pydocstyle-enabled-AD4CD3)](http://www.pydocstyle.org/en/stable/)

[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/opengeokube/kit4dl/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8328176.svg)](https://doi.org/10.5281/zenodo.8328176)
[![pytest](https://github.com/opengeokube/kit4dl/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/opengeokube/kit4dl/actions/workflows/test.yml)

[![PyPI version](https://badge.fury.io/py/kit4dl.svg)](https://badge.fury.io/py/kit4dl)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/kit4dl.svg)](https://anaconda.org/conda-forge/kit4dl) 
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/kit4dl.svg)](https://anaconda.org/conda-forge/kit4dl)




**Kit4DL** is a Python framework for simple, fast, and customisable deep learning prototyping and development. It uses *PyTorch Lightning* as a backend, since it is a powerful, yet quick-to-set-up, framework for deep learning pipelines management, and it relies on TOML (TOML format documentation can be found in [https://toml.io/](https://toml.io/)) configuration file. Though TOML is a relatively new format, it is gaining more and more attention in the Python ecosystem thanks to its intuitiveness, conciseness, and ease of reading. Furthermore, *Kit4DL* uses *torchmetrics* library for *PyTorch*-integrated and efficient implementation of a miscellaneous of metrics.

## 🖋️ Authors
1. Jakub Walczak <a href="https://orcid.org/0000-0002-5632-9484"><img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16" /></a>

1. Marco Macini <a href="https://orcid.org/0000-0002-9150-943X"><img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16" /></a>

1. Mirko Stojiljkovic <a href="https://orcid.org/0000-0003-2256-1645"><img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16" /></a>

1. Shahbaz Alvi <a href="https://orcid.org/0000-0001-5779-8568"><img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16" /></a>


## 🙏 Acknowledgement
This work has received fundings from the Polish National Centre for Research and Development under the LIDER XI program [grant number 0092/L-11/2019, "Semantic analysis of 3D point clouds"] and from the European Union’s Horizon 2020 Research and Innovation programme [SILVANUS Project - grant agreement number 101037247].

## 📜 Cite Us
``` { .bibtex .copy }
@SOFTWARE{kit4dl,
  author = {Walczak, Jakub and
            Mancini, Marco and
            Stojiljković, Mirko and
            Alvi, Shahbaz},
  title = {Kit4DL},
  month = sep,
  year = 2023,
  note = {{Available in GitHub: https://github.com/opengeokube/kit4dl}},
  publisher = {Zenodo},
  version = {2023.9b1},
  doi = {10.5281/zenodo.8328176},
  url = {https://doi.org/10.5281/zenodo.8328176}
}
```