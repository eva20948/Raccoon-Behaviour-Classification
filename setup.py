from setuptools import find_packages, setup

setup(
    name='raccoon_acc_setup',
    packages=find_packages(include=['mypythonlib']),
    version='0.1.0',
    description='Library for the usage of acceleration data',
    author='Me',
    install_requires = ['pandas', 'numpy', 'matplotlib']
)