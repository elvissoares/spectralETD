# spectralETD

**spectralETD** is a Python library which combines Exponential Time Differencing and Pseudo-spectral Methods for Phase-Field Model Equation in three dimensions in a GPU-accelerated framework.

## Requirements

* [Python](https://www.python.org/) 3.10 or later
* [NumPy](https://numpy.org) is the fundamental package for scientific computing with Python.
* [SciPy](https://scipy.org/) is a collection of fundamental algorithms for scientific computing in Python.
* [Matplotlib](https://matplotlib.org/stable/index.html) is a comprehensive library for creating static, animated, and interactive visualizations in Python.
* [PyTorch](https://pytorch.org/) is a high-level library for machine learning, with multidimensional tensors that can also be operated on a CUDA-capable NVIDIA GPU. 

### Optional
* [SciencePlots](https://github.com/garrettj403/SciencePlots) is a Matplotlib styles complement for scientific figures

## Time Integration Methods

* `'IMEX'`: implicit-explicit Euler method;
* `'IF'`: Integrating Factor method
* `'ETD'`: Exponential Time Differencing

## Installation

The easiest way to install spectralETD is by using `pip`:

```Shell
pip install spectralETD
```

To install the development version:
```Shell
pip install git+https://github.com/elvissoares/spectralETD.git
```

or 

```Shell
git clone https://github.com/elvissoares/spectralETD
cd spectralETD
pip install -e .
```

Check out the [examples](examples.md) section for further information.

## Citing spectralETD

If you use spectralETD in your work, please consider to cite it using the following reference:

Soares, E. do A., Barreto, A. G. & Tavares, F. W. *Exponential Integrators for Phase-Field Equations using Pseudo-spectral Methods: A Python Implementation.* 1–12 (2023). ArXiv: [2305.08998](http://arxiv.org/abs/2305.08998)

Bibtex:

    @article{Soares2023,
    archivePrefix = {arXiv},
    arxivId = {2305.08998},
    author = {Soares, Elvis do A. and Barreto, Amaro G. and Tavares, Frederico W},
    eprint = {2305.08998},
    month = {may},
    pages = {1--12},
    title = {{Exponential Integrators for Phase-Field Equations using Pseudo-spectral Methods: A Python Implementation}},
    url = {http://arxiv.org/abs/2305.08998},
    year = {2023}
    }


## Contact
Prof. Elvis do A. Soares

e-mail: elvis@peq.coppe.ufrj.br

Chemical Engineering Program - COPPE

Universidade Federal do Rio de janeiro (UFRJ)

Brazil 