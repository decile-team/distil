from setuptools import setup
import setuptools

setup(
    name='decile-distil',
    version='0.2.0',
    author='Apurva Dani, Durga Sivasubramanian, Nathan Beck, Rishabh Iyer',
    author_email='apurvadani98@gmail.com',
    url='https://github.com/decile-team/distil',
    download_url = 'https://github.com/decile-team/distil/archive/refs/tags/0.2.0.tar.gz',
    license='LICENSE',
    packages=setuptools.find_packages(),
    description='DISTIL is a package for Deep dIverSified inTeractIve Learning.',
    install_requires=[
        "numpy >= 1.14.2",
        "scipy >= 1.0.0",
        "numba >= 0.43.0",
        "tqdm >= 4.24.0",
        "torch >= 1.4.0",
        "submodlib >= 1.1.2",
        "scikit-learn == 0.23.0",
        "multipledispatch == 0.6.0",
        "pandas",
        "torchvision"
    ],
)