import setuptools
# This is a setup script for the spectralETD package, which implements a combination of Exponential Time Differencing and Pseudo-spectral Methods for Phase-Field Model Equation.
setuptools.setup(
    name="spectralETD",
    version="1.0",
    author="Prof. Elvis Soares", #<<<
    author_email="elvis@peq.coppe.ufrj.br", #<<<
    description="A Python Implementation of combining Exponential Time Differencing and Pseudo-spectral Methods for Phase-Field Model Equation", #<<<
    url="https://github.com/elvissoares/spectralETD",  #<<<
    python_requires=">=3.0",  #<<<
    install_requires=[         
        'pandas',         
        'numpy',
        'matplotlib',
        'scipy',
        'scienceplots',
        'torch >= 2.0.0',  # Ensure PyTorch is installed with CUDA 11.8
    ],  #<<<
    packages=setuptools.find_packages(
        where='src',  # Specify the source directory
        include=['spectralETD*','volumerender*'],  # alternatively: `exclude=['additional*']`
        ), #<<<
    package_dir={"": "src"}, # Specify the package directory
    classifiers=[
        "Programming Language :: Python :: 3", 
        "Operating System :: OS Independent",
    ],
)
