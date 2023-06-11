# spectralETD
A Python Implementation of combining Exponential Time Differencing and Pseudo-spectral Methods for Phase-Field Model Equation

## Dependencies

* [NumPy](https://numpy.org) is the fundamental package for scientific computing with Python.
* [SciPy](https://scipy.org/) is a collection of fundamental algorithms for scientific computing in Python.
* [Matplotlib](https://matplotlib.org/stable/index.html) is a comprehensive library for creating static, animated, and interactive visualizations in Python.
* *Optional*: [SciencePlots](https://github.com/garrettj403/SciencePlots) is a Matplotlib styles complement for scientific figures

## Time Integration Methods

* `'IMEX'`: implicit-explicit Euler method;
* `'IF'`: Integrating Factor method
* `'ETD'`: Exponential Time Differencing

## Installation

Clone `spectralETD` repository if you haven't done it yet.

```Shell
git clone https://github.com/elvissoares/spectralETD
```

Go to `spectralETD`'s folder and run any `.ipynb` file as a Jupyter Notebook.

## Examples

* 1D Burgers equations;
* 1D Advection-Diffusion equation;
* 2D Cahn-Hilliard equation;
* 2D Phase-Field Crystal equation;

You can see some movies of the examples in the ``movies`` folder.

# Cite SpectralETD

If you use SpectralETD in your work, please consider to cite it using the following reference:

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


# Contact
Elvis Soares: elvis.asoares@gmail.com

Universidade Federal do Rio de janeiro

School of Chemistry