import sys
sys.path.append('./src')
sys.path.append('./test')

from setuptools import setup, find_packages

setup(
    name='stackly',
    packages=find_packages(),
    test_suite='test.make_suite',
)
