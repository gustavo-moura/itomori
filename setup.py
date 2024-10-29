from setuptools import setup, find_packages

setup(
    name='itomori',
    version='1.0.0',
    url='https://github.com/gustavo-moura/itomori/tree/main',
    author='Gustavo de Moura',
    author_email='gustavo.closure811@passmail.com',
    description='Chance Constraint Markov Decision Process',
    packages=find_packages(),    
    install_requires=['numpy >= 1.11.1', 'matplotlib >= 1.5.1'],
)

setup(
    name="itomori",
    version="0.0.1",
    install_requires=["gymnasium==0.26.0", "pygame==2.1.0"],
)