# src/spectralETD/__init__.py

"""
spectraletd: a Python library which combines Exponential Time Differencing and Pseudo-spectral Methods for Phase-Field Model Equation in three dimensions in a GPU-accelerated framework.
"""

__author__  = "Elvis Soares <elvis.asoares@gmail.com>"
__license__ = "MIT"

from .spectralETD import SpectralETD
from .volumerender import volumerender

__all__ = ["SpectralETD","volumerender"]