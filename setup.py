from setuptools import setup
import setuptools

setup(
    name='distil',
    version='0.0.1',
    author='Apurva Dani, Durga Sivasubramanian, Rishabh Iyer',
    author_email='apurvadani98@gmail.com',
    url='https://github.com/decile-team/distil',
    download_url = 'https://github.com/decile-team/distil/archive/0.0.1.tar.gz',
    license='LICENSE',
    packages=setuptools.find_packages(),
    description='distil is a package for Deep dIverSified inTeractIve Learning.',
    install_requires=[
        "numpy >= 1.14.2",
        "scipy >= 1.0.0",
        "numba >= 0.43.0",
        "tqdm >= 4.24.0",
        "torch >= 1.4.0",
        "apricot-select >= 0.6.0"
        "matplotlib >= 3.3.3"
        "multipledispatch >=0.6.0"
        "scikit-learn >=0.23.0"
    ],
)