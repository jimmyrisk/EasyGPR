from setuptools import setup, find_packages

setup(
    name='EasyGPR',
    version='0.1.0',
    description='A package for easy Gaussian Process Regression',
    author='Jimmy Risk',
    author_email='jrisk@cpp.edu',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'torch',
        'gpytorch',
    ],
    license='LICENSE',
    long_description=open('readme.md').read(),
    url='https://github.com/jimmyrisk/EasyGPR',
)
