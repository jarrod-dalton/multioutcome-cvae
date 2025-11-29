from setuptools import setup, find_packages

setup(
    name="multibin_cvae",
    version="0.0.1",
    description="Conditional VAE for multivariate binary outcomes",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
    ],
)
