from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    version='0.1.0',
    description='Personal project that attempts to incrementally build a nice modular deep learning framework from scratch for learning pruposes.',
    author='Charlesaraya',
    license='MIT',
)